# A04 多空博弈因子 - 高性能优化版 v2.2（方案B：5分钟价格收益率 + 修复版）
# 来源：方正证券《股票日内多空博弈激烈程度度量与"多空博弈"因子构建》（2023-11-29）
# 档案编号：FQ-20260403-014
# 版本：v2.2 - 修复均值距离化冗余步骤 + 异常处理 + README文字
#
# 【v2.2修复内容】
#   1. 均值距离化：去掉对zscore再减截面均值的冗余步骤（zscore_mean≈0，直接abs即可）
#   2. 异常处理：parallel_process_batch_v21() 中 pass → print(f"股票 {inst} 处理失败: {e}")
#   3. README同步：剔除收盘前3分钟 → 剔除收盘前4分钟（与SQL 14:56一致）
#
# 【方案B修改说明】
# 原v2.0版本：过去5分钟收益率 = sum(最近5个1分钟收益率)
# 本v2.1/v2.2版本：过去5分钟收益率 = close_t / close_{t-5} - 1（价格比率法）
#
# 因子构建逻辑：
#   1. 成交量博弈-收益率因子：
#      - 将每分钟成交量按过去5分钟收益率排序
#      - 正序成交量 - 倒序成交量
#   2. 振幅博弈因子：
#      - 振幅 = (最高价 - 最低价) / 收盘价
#      - 类似成交量博弈的计算方法
#   3. 多空博弈因子 = (成交量博弈 + 振幅博弈) / 2
#
# 优化点：
#   - 多进程并行计算（8核）
#   - 批量SQL查询（减少IO）
#   - 向量化滚动计算（无Python循环）

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A04"                # 本因子编号（多空博弈）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
RET_WINDOW = 5                     # 收益率计算窗口（5分钟）

# ========== 性能配置 ==========
MAX_WORKERS = 8                    # 并行进程数
CHUNK_SIZE = 1000                  # 批量处理大小


# ========== 高性能工具函数 ==========

def get_last_computed_date(table_id: str, factor_name: str = FACTOR_NAME) -> Optional[str]:
    """获取该因子最后计算的日期（优化版：使用SQL直接查询）"""
    try:
        sql = f"""
        SELECT MAX(date) as max_date 
        FROM {table_id} 
        WHERE {factor_name} IS NOT NULL
        """
        q = dai.query(sql)
        result = q.df()
        
        if not result.empty and result['max_date'].iloc[0] is not None:
            last_date = pd.to_datetime(result['max_date'].iloc[0])
            extended_date = last_date - timedelta(days=SAFETY_BUFFER_DAYS)
            result_str = extended_date.strftime('%Y-%m-%d')
            print(f"检测到历史数据，最后计算日期: {last_date.date()}，增量起始: {result_str}")
            return result_str
    except Exception as e:
        print(f"读取历史数据失败: {e}")
    return None


def prepare_factor_df_for_write(df: pd.DataFrame, factor_name: str = FACTOR_NAME) -> pd.DataFrame:
    """规范因子DataFrame（优化版）"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    if 'bull_bear_factor' in df.columns:
        df.rename(columns={'bull_bear_factor': factor_name}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['instrument'] = df['instrument'].astype(str).str.strip()
    df[factor_name] = pd.to_numeric(df[factor_name], errors='coerce')
    
    df_out = df[['date', 'instrument', factor_name]].dropna()
    return df_out.drop_duplicates(subset=['date', 'instrument'], keep='last').reset_index(drop=True)


def safe_write_to_library(df: pd.DataFrame, 
                          table_id: str = FACTOR_LIBRARY_TABLE, 
                          factor_name: str = FACTOR_NAME, 
                          overwrite: bool = False) -> int:
    """将因子数据写入宽表"""
    if df is None or df.empty:
        print("没有数据可以写入。")
        return 0
    
    try:
        try:
            ds = dai.DataSource(table_id)
            existing_df = ds.read_bdb(as_type=pd.DataFrame)
            table_exists = True
        except:
            table_exists = False
            existing_df = pd.DataFrame()
        
        if table_exists and factor_name not in existing_df.columns:
            def add_column(df):
                df[factor_name] = np.nan
                return df
            ds.apply_bdb(add_column, as_type=pd.DataFrame)
            existing_df = ds.read_bdb(as_type=pd.DataFrame)
        
        if overwrite and table_exists and factor_name in existing_df.columns:
            def clear_column(df):
                df[factor_name] = np.nan
                return df
            ds.apply_bdb(clear_column, as_type=pd.DataFrame)
            existing_df = ds.read_bdb(as_type=pd.DataFrame)
        
        if table_exists and not existing_df.empty:
            df_indexed = df.set_index(['date', 'instrument'])
            existing_indexed = existing_df.set_index(['date', 'instrument'])
            existing_indexed.update(df_indexed[[factor_name]])
            
            new_keys = df_indexed.index.difference(existing_indexed.index)
            if len(new_keys) > 0:
                new_rows = df_indexed.loc[new_keys]
                for col in existing_indexed.columns:
                    if col not in new_rows.columns:
                        new_rows[col] = np.nan
                existing_indexed = pd.concat([existing_indexed, new_rows])
            
            df_combined = existing_indexed.reset_index()
        else:
            df_combined = df
        
        dai.DataSource.write_bdb(
            data=df_combined,
            id=table_id,
            unique_together=["date", "instrument"],
            on_duplicates="last"
        )
        print(f"成功写入 '{factor_name}'：{len(df)} 行")
        return len(df)
    except Exception as e:
        print(f"写入失败: {e}")
        return 0


def fetch_all_a_share_instruments() -> List[str]:
    """获取所有A股代码"""
    sql = "SELECT DISTINCT instrument FROM cn_stock_instruments"
    q = dai.query(sql, full_db_scan=True)
    df = q.df()
    if df is None or df.empty:
        return []
    return df['instrument'].astype(str).tolist()


def chunk_list(lst: List, size: int):
    """列表分块生成器"""
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def fetch_stock_minute_data_batch(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """批量读取分钟级数据（高性能版）"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    
    sql = f"""
    SELECT 
        date,
        open,
        high,
        low,
        close,
        volume,
        amount,
        instrument
    FROM cn_stock_bar1m
    WHERE date >= TIMESTAMP '{start_date} 09:30:00'
      AND date <= TIMESTAMP '{end_date} 15:00:00'
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) >= 935
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) <= 1456
      AND instrument IN ('{instrument_list}')
    """
    
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df['trade_date'] = df['date'].dt.date
        df['instrument'] = df['instrument'].astype(str)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"SQL查询失败: {e}")
        return pd.DataFrame()


# ========== 因子计算函数（核心优化：向量化+并行） ==========

def process_single_stock_optimized_v21(inst: str, df_min: pd.DataFrame, 
                                        ret_window: int = RET_WINDOW) -> Optional[pd.DataFrame]:
    """
    单只股票处理（向量化优化）- v2.1方案B版本
    
    【方案B修改】过去5分钟收益率计算方式：
    - v2.0: rolling sum of 1-minute returns
    - v2.1: close_t / close_{t-5} - 1 (价格比率法)
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < 10:
            return None
        
        df_min = df_min.sort_values('date').copy()
        
        # 按日期分组处理
        daily_results = []
        
        for trade_date, day_df in df_min.groupby('trade_date'):
            if len(day_df) < 10:  # 数据不足
                continue
            
            day_df = day_df.sort_values('date').reset_index(drop=True)
            
            # 【方案B核心修改】计算过去ret_window分钟收益率（价格比率法）
            # 公式: past_return = close_t / close_{t-ret_window} - 1
            day_df['past_return'] = day_df['close'] / day_df['close'].shift(ret_window) - 1
            
            # 删除NaN值（前ret_window个数据点没有足够历史）
            valid_df = day_df[day_df['past_return'].notna()].copy()
            
            if len(valid_df) < 5:
                continue
            
            # 获取过去收益率、成交量、振幅数据
            past_returns = valid_df['past_return'].values
            volumes = valid_df['volume'].values
            
            # 计算振幅 = (最高价 - 最低价) / 收盘价
            amplitudes = (valid_df['high'].values - valid_df['low'].values) / valid_df['close'].values
            
            # 按收益率从小到大排序的索引
            asc_idx = np.argsort(past_returns)
            # 按收益率从大到小排序的索引
            desc_idx = asc_idx[::-1]
            
            # ===== 成交量博弈因子 =====
            # 正序成交量之和（收益率从小到大）
            vol_asc_sum = np.sum(volumes[asc_idx])
            # 倒序成交量之和（收益率从大到小）
            vol_desc_sum = np.sum(volumes[desc_idx])
            # 日频因子 = 正序成交量 - 倒序成交量
            volume_game_daily = vol_asc_sum - vol_desc_sum
            
            # ===== 振幅博弈因子 =====
            # 正序振幅之和（收益率从小到大）
            amp_asc_sum = np.sum(amplitudes[asc_idx])
            # 倒序振幅之和（收益率从大到小）
            amp_desc_sum = np.sum(amplitudes[desc_idx])
            # 日频因子 = 正序振幅 - 倒序振幅
            amplitude_game_daily = amp_asc_sum - amp_desc_sum
            
            daily_results.append({
                'date': pd.to_datetime(trade_date),
                'instrument': inst,
                'volume_game_daily': volume_game_daily,
                'amplitude_game_daily': amplitude_game_daily
            })
        
        if not daily_results:
            return None
        
        return pd.DataFrame(daily_results)
        
    except Exception as e:
        return None


def parallel_process_batch_v21(stock_data_dict: Dict[str, pd.DataFrame], 
                                max_workers: int = MAX_WORKERS) -> List[pd.DataFrame]:
    """并行处理一批股票（v2.1版本）"""
    if not stock_data_dict:
        return []
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_inst = {
            executor.submit(process_single_stock_optimized_v21, inst, df): inst 
            for inst, df in stock_data_dict.items()
        }
        
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                print(f"股票 {inst} 处理失败: {e}")
    
    return results


def calculate_daily_factor_optimized_v21(start_date: str, end_date: str, 
                                          batch_size: int = CHUNK_SIZE,
                                          use_parallel: bool = True) -> pd.DataFrame:
    """
    高性能日频因子计算（成交量博弈 + 振幅博弈）- v2.1方案B版本
    """
    print("获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        return pd.DataFrame()
    
    print(f"获取到 {len(instruments_all)} 只股票，批次大小={batch_size}，并行={use_parallel}")
    print(f"【方案B】5分钟收益率使用价格比率法: close_t / close_{{t-{RET_WINDOW}}} - 1")
    
    all_daily_factors = []
    total_batches = (len(instruments_all) + batch_size - 1) // batch_size
    
    for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
        print(f"\n处理第 {batch_idx}/{total_batches} 批：{len(batch)} 只股票 ...")
        batch_start = datetime.now()
        
        # 批量获取数据
        df_minute_all = fetch_stock_minute_data_batch(batch, start_date, end_date)
        
        if df_minute_all is None or df_minute_all.empty:
            continue
        
        # 分组为每只股票的数据
        stock_groups = {
            inst: group.reset_index(drop=True) 
            for inst, group in df_minute_all.groupby('instrument')
        }
        
        # 并行或串行处理
        if use_parallel and len(stock_groups) > 1:
            batch_results = parallel_process_batch_v21(stock_groups, MAX_WORKERS)
        else:
            batch_results = []
            for inst, df in stock_groups.items():
                result = process_single_stock_optimized_v21(inst, df)
                if result is not None and not result.empty:
                    batch_results.append(result)
        
        if batch_results:
            df_batch = pd.concat(batch_results, axis=0, ignore_index=True)
            all_daily_factors.append(df_batch)
            batch_time = (datetime.now() - batch_start).total_seconds()
            print(f"第 {batch_idx} 批完成：{len(df_batch)} 行，耗时 {batch_time:.2f}秒")
    
    if not all_daily_factors:
        return pd.DataFrame()
    
    df_all = pd.concat(all_daily_factors, axis=0, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    return df_all.drop_duplicates(subset=['date', 'instrument'], keep='last').sort_values(['date', 'instrument'])


def mean_distance_normalize(df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """
    均值距离化处理：截面标准化后减去均值并取绝对值（向量化）
    """
    df = df.copy()
    
    # 截面标准化（z-score）- 向量化
    date_stats = df.groupby('date')[factor_col].agg(['mean', 'std'])
    date_stats.columns = [f'{factor_col}_mean', f'{factor_col}_std']  # 重命名列
    df = df.merge(date_stats, left_on='date', right_index=True, how='left')
    
    df[factor_col + '_zscore'] = np.where(
        df[f'{factor_col}_std'] > 0,
        (df[factor_col] - df[f'{factor_col}_mean']) / df[f'{factor_col}_std'],
        0
    )
    
    # 【v2.2修复】均值距离化：直接对zscore取绝对值（zscore截面均值≈0，无需再减均值）
    df[factor_col + '_norm'] = df[factor_col + '_zscore'].abs()
    
    return df


def calculate_bull_bear_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算多空博弈因子（向量化优化版）
    
    合成公式：
    1) 成交量博弈因子 = (月均成交量博弈 + 月稳成交量博弈) / 2
    2) 振幅博弈因子 = (月均振幅博弈 + 月稳振幅博弈) / 2
    3) 多空博弈因子 = (成交量博弈 + 振幅博弈) / 2
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只，日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 对日频因子进行均值距离化处理（向量化）
    df = mean_distance_normalize(df, 'volume_game_daily')
    df = mean_distance_normalize(df, 'amplitude_game_daily')
    
    # 向量化滚动计算
    min_periods = min(10, FACTOR_WINDOW)
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    # 成交量博弈因子：20日滚动均值和标准差
    df['volume_game_mean'] = df.groupby('instrument')['volume_game_daily_norm'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    df['volume_game_std'] = df.groupby('instrument')['volume_game_daily_norm'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
    )
    
    # 振幅博弈因子：20日滚动均值和标准差
    df['amplitude_game_mean'] = df.groupby('instrument')['amplitude_game_daily_norm'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    df['amplitude_game_std'] = df.groupby('instrument')['amplitude_game_daily_norm'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
    )
    
    # 合成因子（向量化）
    df['volume_game'] = (df['volume_game_mean'] + df['volume_game_std']) / 2
    df['amplitude_game'] = (df['amplitude_game_mean'] + df['amplitude_game_std']) / 2
    df['bull_bear_factor'] = (df['volume_game'] + df['amplitude_game']) / 2
    
    # 只保留有效因子值
    df_result = df[df['bull_bear_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    
    return df_result[['date', 'instrument', 'bull_bear_factor']]


# ========== 主流程 ==========
if __name__ == "__main__":
    import time
    start_time_total = time.time()
    
    # 参数配置
    start_date = "2026-02-04"
    end_date = "2026-03-14"
    overwrite = False
    use_incremental = True
    use_parallel = True
    
    print(f"=== {FACTOR_NAME}（多空博弈）因子计算 - v2.1方案B版本 ===")
    print(f"【版本说明】5分钟收益率使用价格比率法: close_t / close_{{t-{RET_WINDOW}}} - 1")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日, 收益率窗口: {RET_WINDOW}分钟")
    
    # 增量计算
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            buffer_days = FACTOR_WINDOW + 10  # 20日窗口 + 10天缓冲
            effective_start = (pd.to_datetime(last_computed) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"【增量模式】检测到最后计算日期: {last_computed}")
            print(f"【增量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
        else:
            buffer_days = FACTOR_WINDOW + 10
            effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    else:
        buffer_days = FACTOR_WINDOW + 10
        effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
        print(f"【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    
    # 步骤1：计算日频因子
    print("\n【步骤1】计算日频因子（成交量博弈 + 振幅博弈）...")
    start_time_calc = datetime.now()
    df_daily = calculate_daily_factor_optimized_v21(effective_start, end_date, 
                                                     batch_size=CHUNK_SIZE,
                                                     use_parallel=use_parallel)
    calc_time = (datetime.now() - start_time_calc).total_seconds()
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        
        # 步骤2：计算多空博弈因子
        print("\n【步骤2】计算多空博弈因子（向量化滚动）...")
        start_time_factor = datetime.now()
        df_factor = calculate_bull_bear_factor_optimized(df_daily)
        factor_time = (datetime.now() - start_time_factor).total_seconds()
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日频数据列: {df_daily.columns.tolist()}")
            print(f"  建议检查：")
            print(f"    1. 日期范围是否足够（至少需要{FACTOR_WINDOW}个交易日）")
            print(f"    2. 尝试使用更大的日期范围（如3个月以上）")
        else:
            print(f"因子计算完成: {len(df_factor)} 条，耗时 {factor_time:.2f}秒")
            
            # 过滤到指定日期范围
            df_factor['date'] = pd.to_datetime(df_factor['date'])
            original_count = len(df_factor)
            df_factor = df_factor[
                (df_factor['date'] >= pd.to_datetime(start_date)) & 
                (df_factor['date'] <= pd.to_datetime(end_date))
            ].reset_index(drop=True)
            
            print(f"\n日期过滤: {original_count} -> {len(df_factor)} 条")
            
            if not df_factor.empty:
                # 写入数据
                df_to_write = prepare_factor_df_for_write(df_factor, FACTOR_NAME)
                
                start_time_write = datetime.now()
                written_count = safe_write_to_library(df_to_write, overwrite=overwrite)
                write_time = (datetime.now() - start_time_write).total_seconds()
                
                total_time = time.time() - start_time_total
                print(f"\n=== 完成 ===")
                print(f"日频计算: {calc_time:.2f}秒")
                print(f"因子合成: {factor_time:.2f}秒")
                print(f"数据写入: {write_time:.2f}秒")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"写入 {written_count} 条数据")
                print(f"\n预期表现: Rank IC ~ -9.73%（研报基准）")
                print(f"多空年化收益 ~ 40.12%")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")
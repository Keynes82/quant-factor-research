# A01 适度冒险因子 - 高性能优化版（8核并行）
# 来源：方正证券《成交量激增时刻蕴含的alpha信息》（2022-04-12）
# 档案编号：FQ-20260318-009
# 版本：v2.0 - 基于高性能模板（多进程并行优化）
#
# 因子构建逻辑：
#   1. 定义成交量激增时刻：分钟成交量增加量 > mean + std
#   2. 定义"耀眼5分钟"：激增时刻及其随后4分钟
#   3. 计算"耀眼波动率"：耀眼5分钟内收益率的标准差
#   4. 计算"耀眼收益率"：激增时刻的分钟收益率
#   5. 构建"适度"指标：|个股值 - 截面均值|
#   6. 合成"适度冒险"因子：波动率维度 + 收益率维度
#
# 优化点：
#   - 多进程并行计算（8核）
#   - 批量SQL查询（减少IO）
#   - 向量化滚动计算（无Python循环）

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A01"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口
DAZZLING_MINUTES = 5               # 耀眼N分钟

# ========== 性能配置 ==========
MAX_WORKERS = 8                    # 并行进程数
CHUNK_SIZE = 1000                  # 批量处理大小


# ========== 高性能工具函数 ==========

def get_last_computed_date(table_id: str, factor_name: str = FACTOR_NAME) -> Optional[str]:
    """获取该因子最后计算的日期（优化版）"""
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
    
    if 'moderate_risk_factor' in df.columns:
        df.rename(columns={'moderate_risk_factor': factor_name}, inplace=True)
    
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

def identify_surge_minutes_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    向量化识别成交量激增时刻
    
    优化点：使用groupby.transform避免逐行循环
    """
    df = df.sort_values(['instrument', 'trade_date', 'date']).copy()
    
    # 计算成交量增加量（向量化）
    df['volume_diff'] = df.groupby(['instrument', 'trade_date'])['volume'].diff()
    
    # 计算每组的mean和std（向量化transform）
    df['volume_diff_mean'] = df.groupby(['instrument', 'trade_date'])['volume_diff'].transform('mean')
    df['volume_diff_std'] = df.groupby(['instrument', 'trade_date'])['volume_diff'].transform('std')
    
    # 识别激增时刻（向量化）
    df['is_surge'] = df['volume_diff'] > (df['volume_diff_mean'] + df['volume_diff_std'])
    
    # 计算分钟收益率（向量化）
    df['return'] = df.groupby(['instrument', 'trade_date'])['close'].pct_change()
    
    return df


def calculate_dazzling_metrics_vectorized(df: pd.DataFrame, dazzling_minutes: int = DAZZLING_MINUTES) -> pd.DataFrame:
    """
    向量化计算耀眼波动率和收益率
    
    优化点：使用rolling窗口和groupby避免逐行处理
    """
    # 为每个激增时刻标记组号
    df['surge_group'] = df.groupby(['instrument', 'trade_date'])['is_surge'].cumsum()
    
    # 只保留激增时刻开始的数据
    df_surge = df[df['is_surge']].copy()
    
    if df_surge.empty:
        return pd.DataFrame()
    
    # 对每个激增时刻，计算随后dazzling_minutes的统计量
    results = []
    
    for (inst, trade_date), group in df_surge.groupby(['instrument', 'trade_date']):
        # 获取该股票该日期的完整数据
        day_data = df[(df['instrument'] == inst) & (df['trade_date'] == trade_date)].copy()
        day_data = day_data.reset_index(drop=True)
        
        surge_indices = day_data[day_data['is_surge']].index.tolist()
        
        dazzling_vols = []
        dazzling_rets = []
        
        for idx in surge_indices:
            end_idx = min(idx + dazzling_minutes, len(day_data))
            window = day_data.iloc[idx:end_idx]
            
            if len(window) >= 2:
                vol = window['return'].std()
                ret = window['return'].iloc[0] if not window.empty else np.nan
                
                if not np.isnan(vol):
                    dazzling_vols.append(vol)
                if not np.isnan(ret):
                    dazzling_rets.append(ret)
        
        if dazzling_vols and dazzling_rets:
            results.append({
                'instrument': inst,
                'trade_date': trade_date,
                'daily_dazzling_volatility': np.mean(dazzling_vols),
                'daily_dazzling_return': np.mean(dazzling_rets)
            })
    
    return pd.DataFrame(results)


def process_single_stock_optimized(inst: str, df_min: pd.DataFrame, 
                                   dazzling_minutes: int = DAZZLING_MINUTES) -> Optional[pd.DataFrame]:
    """
    单只股票处理（向量化优化）- 修复：处理多日期数据
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < 10:
            return None
        
        df_min = df_min.sort_values('date').copy()
        
        # 修复：按日期分组处理，确保每个交易日都有数据
        daily_results = []
        
        for trade_date, day_df in df_min.groupby('trade_date'):
            if len(day_df) < 5:
                continue
            
            day_df = day_df.sort_values('date').reset_index(drop=True)
            
            # 计算成交量增加量
            day_df['volume_diff'] = day_df['volume'].diff()
            volume_diff_mean = day_df['volume_diff'].mean()
            volume_diff_std = day_df['volume_diff'].std()
            
            if pd.isna(volume_diff_mean) or pd.isna(volume_diff_std) or volume_diff_std == 0:
                continue
                
            threshold = volume_diff_mean + volume_diff_std
            
            # 识别激增时刻
            day_df['is_surge'] = day_df['volume_diff'] > threshold
            day_df['return'] = day_df['close'].pct_change()
            
            surge_indices = day_df[day_df['is_surge']].index.tolist()
            
            if not surge_indices:
                continue
            
            dazzling_vols = []
            dazzling_rets = []
            
            for idx in surge_indices:
                end_idx = min(idx + dazzling_minutes, len(day_df))
                window = day_df.loc[idx:end_idx-1]
                
                if len(window) >= 2:
                    vol = window['return'].std()
                    ret = day_df.loc[idx, 'return']
                    
                    if not np.isnan(vol):
                        dazzling_vols.append(vol)
                    if not np.isnan(ret):
                        dazzling_rets.append(ret)
            
            if dazzling_vols and dazzling_rets:
                daily_results.append({
                    'date': pd.to_datetime(trade_date),
                    'instrument': inst,
                    'daily_dazzling_volatility': np.mean(dazzling_vols),
                    'daily_dazzling_return': np.mean(dazzling_rets)
                })
        
        if not daily_results:
            return None
        
        return pd.DataFrame(daily_results)
        
    except Exception as e:
        return None


def parallel_process_batch(stock_data_dict: Dict[str, pd.DataFrame], 
                           max_workers: int = MAX_WORKERS) -> List[pd.DataFrame]:
    """并行处理一批股票"""
    if not stock_data_dict:
        return []
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_inst = {
            executor.submit(process_single_stock_optimized, inst, df): inst 
            for inst, df in stock_data_dict.items()
        }
        
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                pass  # 静默处理单个股票错误
    
    return results


def calculate_daily_factor_optimized(start_date: str, end_date: str, 
                                     batch_size: int = CHUNK_SIZE,
                                     use_parallel: bool = True) -> pd.DataFrame:
    """
    高性能日频因子计算
    """
    print("获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        return pd.DataFrame()
    
    print(f"获取到 {len(instruments_all)} 只股票，批次大小={batch_size}，并行={use_parallel}")
    
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
            batch_results = parallel_process_batch(stock_groups, MAX_WORKERS)
        else:
            batch_results = []
            for inst, df in stock_groups.items():
                # 按日期分组处理
                for trade_date, day_df in df.groupby('trade_date'):
                    result = process_single_stock_optimized(inst, day_df)
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


def calculate_moderate_risk_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算适度冒险因子（向量化优化版）
    
    修复：min_periods改为10（允许部分数据计算）
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只，日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 计算截面均值（向量化）
    date_stats = df.groupby('date').agg({
        'daily_dazzling_volatility': 'mean',
        'daily_dazzling_return': 'mean'
    }).rename(columns={
        'daily_dazzling_volatility': 'vol_mean',
        'daily_dazzling_return': 'ret_mean'
    })
    
    df = df.merge(date_stats, left_on='date', right_index=True, how='left')
    
    # 计算适度指标（向量化）
    df['moderate_vol'] = (df['daily_dazzling_volatility'] - df['vol_mean']).abs()
    df['moderate_ret'] = (df['daily_dazzling_return'] - df['ret_mean']).abs()
    
    # 计算每日适度指标（向量化）
    # 研报原文："将波动率维度和收益率维度的适度指标取均值"
    df['daily_moderate'] = (df['moderate_vol'] + df['moderate_ret']) / 2
    
    # 向量化滚动平滑（20日滚动均值）
    min_periods = min(10, FACTOR_WINDOW)
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    df['moderate_risk_factor'] = df.groupby('instrument')['daily_moderate'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 清理中间列
    df_result = df[['date', 'instrument', 'moderate_risk_factor']].dropna()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    
    return df_result


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
    
    print(f"=== {FACTOR_NAME}（适度冒险）因子计算 - 高性能优化版 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日, 耀眼分钟: {DAZZLING_MINUTES}")
    
    # 增量计算
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            # 修复：增量起始日期需要往前推足够天数（FACTOR_WINDOW + 缓冲），确保滚动窗口有数据
            buffer_days = FACTOR_WINDOW + 10  # 20日窗口 + 10天缓冲
            effective_start = (pd.to_datetime(last_computed) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"【增量模式】检测到最后计算日期: {last_computed}")
            print(f"【增量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
        else:
            # 全量模式：也需要往前推足够天数
            buffer_days = FACTOR_WINDOW + 10
            effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    else:
        buffer_days = FACTOR_WINDOW + 10
        effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
        print(f"【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    
    # 步骤1：计算日频因子
    print("\n【步骤1】计算日频因子（并行优化）...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor_optimized(effective_start, end_date, 
                                                 batch_size=CHUNK_SIZE,
                                                 use_parallel=use_parallel)
    calc_time = time.time() - start_time_calc
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        
        # 步骤2：计算适度冒险因子
        print("\n【步骤2】计算适度冒险因子（向量化滚动）...")
        start_time_factor = time.time()
        df_factor = calculate_moderate_risk_factor_optimized(df_daily)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日频数据列: {df_daily.columns.tolist()}")
            print(f"  日频数据统计:")
            print(f"    - daily_dazzling_volatility: 均值={df_daily['daily_dazzling_volatility'].mean():.6f}, 非空数={df_daily['daily_dazzling_volatility'].notna().sum()}")
            print(f"    - daily_dazzling_return: 均值={df_daily['daily_dazzling_return'].mean():.6f}, 非空数={df_daily['daily_dazzling_return'].notna().sum()}")
            print(f"  建议检查：")
            print(f"    1. 日期范围是否足够（至少需要{FACTOR_WINDOW}个交易日）")
            print(f"    2. 日频数据是否有空值")
            print(f"    3. 尝试使用更大的日期范围（如3个月以上）")
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
                
                start_time_write = time.time()
                written_count = safe_write_to_library(df_to_write, overwrite=overwrite)
                write_time = time.time() - start_time_write
                
                total_time = time.time() - start_time_total
                print(f"\n=== 完成 ===")
                print(f"日频计算: {calc_time:.2f}秒")
                print(f"因子合成: {factor_time:.2f}秒")
                print(f"数据写入: {write_time:.2f}秒")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"写入 {written_count} 条数据")
                print(f"\n预期表现: Rank IC ~ -8.89%")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")

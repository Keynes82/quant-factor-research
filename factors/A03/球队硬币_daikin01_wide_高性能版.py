# A03 球队硬币因子 - 高性能优化版（8核并行）
# 来源：方正证券《个股动量效应的识别及"球队硬币"因子构建》（2022-06-11）
# 档案编号：FQ-20260318-006
# 版本：v2.0 - 基于高性能模板（多进程并行优化）
#
# 因子构建逻辑：
#   借鉴Moskowitz(2021)的"球队-硬币"理论：
#   - "硬币型"股票（可知性高）：波动率低、换手稳定 → 预期动量 → 收益率×(-1)
#   - "球队型"股票（可知性低）：波动率高、换手变化大 → 预期反转 → 收益率保持不变
#
#   球队硬币因子 = (修正日间反转 + 修正日内反转 + 修正隔夜反转) / 3
#   其中每个修正反转因子 = (波动翻转因子 + 换手翻转因子) / 2
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
FACTOR_NAME = "A03"                # 本因子编号（球队硬币）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）

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
    
    if 'team_coin_factor' in df.columns:
        df.rename(columns={'team_coin_factor': factor_name}, inplace=True)
    
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
            print(f"注意：表 '{table_id}' 中不存在 '{factor_name}' 列，使用 apply_bdb 添加...")
            def add_column(df):
                df[factor_name] = np.nan
                return df
            ds.apply_bdb(add_column, as_type=pd.DataFrame)
            print(f"已添加 '{factor_name}' 列到表 '{table_id}'")
            existing_df = ds.read_bdb(as_type=pd.DataFrame)
        
        if overwrite and table_exists and factor_name in existing_df.columns:
            def clear_column(df):
                df[factor_name] = np.nan
                return df
            ds.apply_bdb(clear_column, as_type=pd.DataFrame)
            print(f"已清除 '{factor_name}' 列的历史数据。")
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


def fetch_stock_daily_data_batch(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """批量读取日频数据（高性能版）"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    start_dt_extended = (pd.to_datetime(start_date) - pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    
    sql = f"""
    SELECT 
        date,
        open,
        close,
        volume,
        amount,
        turn,
        instrument
    FROM cn_stock_bar1d
    WHERE date >= TIMESTAMP '{start_dt_extended} 00:00:00'
      AND date <= TIMESTAMP '{end_date} 23:59:59'
      AND instrument IN ('{instrument_list}')
    """
    
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'close', 'volume', 'amount', 'turn']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['instrument'] = df['instrument'].astype(str)
        
        return df
    except Exception as e:
        print(f"SQL查询失败: {e}")
        return pd.DataFrame()


# ========== 因子计算函数（核心优化：向量化+并行） ==========

def process_single_stock_team_coin(inst: str, df_min: pd.DataFrame, df_daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    单只股票处理（球队硬币因子逻辑）- 向量化优化
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < 10:
            return None
        
        df_min = df_min.sort_values('date').copy()
        
        daily_results = []
        
        for trade_date, day_df in df_min.groupby('trade_date'):
            if len(day_df) < 5:
                continue
            
            day_df = day_df.sort_values('date').reset_index(drop=True)
            
            # 计算日内收益率 = 收盘/开盘 - 1
            daily_open = day_df.iloc[0]['open']
            daily_close = day_df.iloc[-1]['close']
            
            if daily_open <= 0 or pd.isna(daily_open):
                continue
                
            intraday_return = daily_close / daily_open - 1
            
            # 计算分钟收益率波动率
            day_df['minute_return'] = day_df['close'].pct_change()
            intraday_volatility = day_df['minute_return'].std()
            
            if pd.isna(intraday_volatility):
                continue
            
            daily_results.append({
                'date': pd.to_datetime(trade_date),
                'instrument': inst,
                'intraday_return': intraday_return,
                'intraday_volatility': intraday_volatility
            })
        
        if not daily_results:
            return None
        
        df_result = pd.DataFrame(daily_results)
        
        # 合并日频数据（换手率）
        if df_daily is not None and not df_daily.empty:
            df_daily_inst = df_daily[df_daily['instrument'] == inst].copy()
            if not df_daily_inst.empty:
                df_daily_inst['date'] = pd.to_datetime(df_daily_inst['date']).dt.normalize()
                df_result['date'] = pd.to_datetime(df_result['date']).dt.normalize()
                df_result = df_result.merge(df_daily_inst[['date', 'turn']], on='date', how='left')
                # 计算换手率变化量
                df_result['turn_change'] = df_result['turn'].diff()
            else:
                df_result['turn'] = np.nan
                df_result['turn_change'] = np.nan
        else:
            df_result['turn'] = np.nan
            df_result['turn_change'] = np.nan
        
        return df_result
        
    except Exception as e:
        return None


def parallel_process_batch_team_coin(stock_minute_dict: Dict[str, pd.DataFrame],
                                     stock_daily_dict: Dict[str, pd.DataFrame],
                                     max_workers: int = MAX_WORKERS) -> List[pd.DataFrame]:
    """并行处理一批股票（球队硬币专用）"""
    if not stock_minute_dict:
        return []
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_inst = {
            executor.submit(process_single_stock_team_coin, inst, 
                           stock_minute_dict.get(inst, pd.DataFrame()),
                           stock_daily_dict.get(inst, pd.DataFrame())): inst 
            for inst in stock_minute_dict.keys()
        }
        
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                pass
    
    return results


def calculate_daily_factor_optimized(start_date: str, end_date: str,
                                     batch_size: int = CHUNK_SIZE,
                                     use_parallel: bool = True) -> pd.DataFrame:
    """
    高性能日频因子计算（球队硬币）
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
        
        # 批量获取分钟数据
        df_minute_all = fetch_stock_minute_data_batch(batch, start_date, end_date)
        # 批量获取日频数据
        df_daily_all = fetch_stock_daily_data_batch(batch, start_date, end_date)
        
        if df_minute_all is None or df_minute_all.empty:
            continue
        
        # 分组为每只股票的数据
        stock_minute_groups = {
            inst: group.reset_index(drop=True)
            for inst, group in df_minute_all.groupby('instrument')
        }
        
        stock_daily_groups = {}
        if df_daily_all is not None and not df_daily_all.empty:
            stock_daily_groups = {
                inst: group.reset_index(drop=True)
                for inst, group in df_daily_all.groupby('instrument')
            }
        
        # 并行或串行处理
        if use_parallel and len(stock_minute_groups) > 1:
            batch_results = parallel_process_batch_team_coin(stock_minute_groups, stock_daily_groups, MAX_WORKERS)
        else:
            batch_results = []
            for inst, df_min in stock_minute_groups.items():
                df_daily = stock_daily_groups.get(inst, pd.DataFrame())
                result = process_single_stock_team_coin(inst, df_min, df_daily)
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


def calculate_team_coin_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算球队硬币因子（向量化优化版）
    
    因子构建逻辑：
    1. 计算日内收益率均值（月均）和标准差（月稳）
    2. 计算截面均值判断"硬币型"或"球队型"
    3. 波动翻转：波动率<市场均值（硬币型）→ 收益率×(-1)
    4. 换手翻转：换手率变化量<市场均值（硬币型）→ 收益率×(-1)
    5. 修正日内反转因子 = (波动翻转 + 换手翻转) / 2
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只，日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 向量化计算20日滚动指标
    min_periods = min(10, FACTOR_WINDOW)
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    df['intraday_return_mean'] = df.groupby('instrument')['intraday_return'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    df['intraday_return_std'] = df.groupby('instrument')['intraday_return'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
    )
    df['turn_change_mean'] = df.groupby('instrument')['turn_change'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 计算截面均值（向量化）
    date_stats = df.groupby('date').agg({
        'intraday_return_std': 'mean',
        'turn_change': 'mean'
    }).rename(columns={
        'intraday_return_std': 'vol_market_mean',
        'turn_change': 'turn_change_market_mean'
    })
    
    df = df.merge(date_stats, left_on='date', right_index=True, how='left')
    
    # 判断硬币型/球队型（向量化）
    df['is_coin_vol'] = df['intraday_return_std'] < df['vol_market_mean']
    df['is_coin_turn'] = df['turn_change'] < df['turn_change_market_mean']
    
    # 波动翻转因子（向量化）
    df['vol_flip_factor'] = np.where(df['is_coin_vol'],
                                     -df['intraday_return_mean'],
                                     df['intraday_return_mean'])
    
    # 换手翻转因子（向量化）
    df['turn_flip_factor'] = np.where(df['is_coin_turn'],
                                      -df['intraday_return_mean'],
                                      df['intraday_return_mean'])
    
    # 修正日内反转因子 = (波动翻转 + 换手翻转) / 2
    df['revised_intraday_reversal'] = (df['vol_flip_factor'] + df['turn_flip_factor']) / 2
    
    # 球队硬币因子 = 修正日内反转因子（简化版）
    df['team_coin_factor'] = df['revised_intraday_reversal']
    
    # 只保留有效因子值
    df_result = df[df['team_coin_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    
    return df_result[['date', 'instrument', 'team_coin_factor']]


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
    
    print(f"=== {FACTOR_NAME}（球队硬币）因子计算 - 高性能优化版 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    
    # 增量计算
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            buffer_days = FACTOR_WINDOW + 10
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
    
    print("\n【说明】当前版本主要实现修正日内反转因子（使用1分钟K线数据）")
    print("完整球队硬币因子 = (修正日间反转 + 修正日内反转 + 修正隔夜反转) / 3")
    
    # 步骤1：计算日频因子
    print("\n【步骤1】计算日频因子（并行优化）...")
    start_time_calc = datetime.now()
    df_daily = calculate_daily_factor_optimized(effective_start, end_date,
                                                 batch_size=CHUNK_SIZE,
                                                 use_parallel=use_parallel)
    calc_time = (datetime.now() - start_time_calc).total_seconds()
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        
        # 步骤2：计算球队硬币因子
        print("\n【步骤2】计算球队硬币因子（向量化滚动）...")
        start_time_factor = datetime.now()
        df_factor = calculate_team_coin_factor_optimized(df_daily)
        factor_time = (datetime.now() - start_time_factor).total_seconds()
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日频数据列: {df_daily.columns.tolist()}")
            print(f"  日频数据统计:")
            print(f"    - intraday_return: 均值={df_daily['intraday_return'].mean():.6f}, 非空数={df_daily['intraday_return'].notna().sum()}")
            print(f"    - intraday_volatility: 均值={df_daily['intraday_volatility'].mean():.6f}, 非空数={df_daily['intraday_volatility'].notna().sum()}")
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
                print(f"\n预期表现: Rank IC ~ -9.67%")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
                
                print(f"\n【说明】当前版本主要实现修正日内反转因子")
                print("完整球队硬币因子 = (修正日间反转 + 修正日内反转 + 修正隔夜反转) / 3")
            else:
                print("过滤后没有数据需要写入。")

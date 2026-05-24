# A08 勇攀高峰因子 - 高性能优化版 v2.1（修正版 + 修复版）
# 档案编号：FQ-20260407-A08-CORRECTED
# 版本：v2.1 - 修复版（异常处理优化）
#
# 【v2.1修复内容】
#   1. 异常处理优化：parallel_process_batch() 中 except: pass → print(f"股票{inst}失败: {e}")
#
# 【v2.0保留内容】
#   1. 严格遵循研报定义修正版
#   2. 更优波动率：基于5分钟窗口OHLC共20个价格，(标准差/均值)²
#   3. 收益波动比：分钟收益率 / 更优波动率
#   4. 异常高波动识别：更优波动率 >= mean + std
#   5. 协方差计算：cov(收益波动比, 更优波动率) 仅在异常高波动时段
#   6. 勇攀高峰因子 = (20日协方差均值 + 20日协方差标准差) / 2

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
FACTOR_NAME = "A08"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
BETTER_VOL_WINDOW = 5              # 更优波动率计算窗口（5分钟）

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
    
    if 'climbing_peak_factor' in df.columns:
        df.rename(columns={'climbing_peak_factor': factor_name}, inplace=True)
    
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
    """批量读取分钟级数据（高性能版）
    
    数据时间过滤：09:35-14:56（剔除开盘前5分钟和收盘前3分钟，与研报一致）
    """
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    
    # 时间过滤：09:35-14:56（与研报一致，剔除开盘前5分钟和收盘前3分钟）
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
      -- 剔除开盘前5分钟(09:30-09:34)和收盘前3分钟(14:57-15:00)
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) >= 935
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) <= 1456
      AND instrument IN ('{instrument_list}')
    ORDER BY instrument, date
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


# ========== 因子计算函数（核心：严格遵循研报定义） ==========

def calculate_better_volatility(prices: np.ndarray) -> float:
    """
    计算更优波动率（严格按研报定义）
    
    研报定义：第t分钟的更优波动率 = (第t-4到t分钟共20个价格的std / mean)²
    
    参数:
        prices: 5分钟窗口的OHLC价格数组，共20个值
                [t-4_open, t-4_high, t-4_low, t-4_close, ..., t_open, t_high, t_low, t_close]
    
    返回:
        更优波动率值 (std/mean)²
    """
    if len(prices) < 20 or np.any(np.isnan(prices)) or np.any(prices == 0):
        return np.nan
    
    mean_price = np.mean(prices)
    if mean_price == 0:
        return np.nan
    
    std_price = np.std(prices, ddof=1)  # 样本标准差
    better_vol = (std_price / mean_price) ** 2
    
    return better_vol


def calculate_intraday_metrics(day_df: pd.DataFrame, window: int = BETTER_VOL_WINDOW) -> pd.DataFrame:
    """
    计算日内更优波动率和收益波动比（严格按研报定义）
    
    研报步骤：
    1. 计算每分钟的更优波动率（5分钟OHLC窗口）
    2. 计算每分钟的收益波动比 = 分钟收益率 / 更优波动率
    """
    if len(day_df) < window + 1:
        return pd.DataFrame()
    
    day_df = day_df.sort_values('date').reset_index(drop=True)
    
    # 计算分钟收益率
    day_df['minute_return'] = day_df['close'].pct_change()
    
    # 计算更优波动率（5分钟OHLC窗口）
    better_vols = []
    for i in range(len(day_df)):
        if i < window - 1:
            better_vols.append(np.nan)
        else:
            # 获取t-4到t分钟共5分钟的OHLC数据
            window_data = day_df.iloc[i - window + 1:i + 1]
            # 提取20个价格：open, high, low, close for each minute
            prices = []
            for _, row in window_data.iterrows():
                prices.extend([row['open'], row['high'], row['low'], row['close']])
            
            if len(prices) == 20:
                better_vol = calculate_better_volatility(np.array(prices))
                better_vols.append(better_vol)
            else:
                better_vols.append(np.nan)
    
    day_df['better_volatility'] = better_vols
    
    # 计算收益波动比 = 分钟收益率 / 更优波动率
    day_df['return_vol_ratio'] = day_df['minute_return'] / day_df['better_volatility']
    
    return day_df.dropna(subset=['better_volatility', 'return_vol_ratio'])


def calculate_daily_covariance(day_df: pd.DataFrame) -> Optional[float]:
    """
    计算当日协方差（严格按研报定义）
    
    研报步骤：
    3. 找到当日更优波动率 >= mean + std 的部分，作为异常高波动时段
    4. 计算异常高波动时段的收益波动比与更优波动率的协方差
    
    返回:
        当日协方差值（仅在异常高波动时段计算）
    """
    if day_df is None or len(day_df) < 10:
        return None
    
    # 计算更优波动率的均值和标准差
    vol_mean = day_df['better_volatility'].mean()
    vol_std = day_df['better_volatility'].std()
    
    if pd.isna(vol_mean) or pd.isna(vol_std):
        return None
    
    # 识别异常高波动时段：更优波动率 >= mean + std
    high_vol_mask = day_df['better_volatility'] >= (vol_mean + vol_std)
    high_vol_data = day_df[high_vol_mask]
    
    if len(high_vol_data) < 3:  # 需要至少3个点才能计算有意义的协方差
        return None
    
    # 计算协方差：cov(收益波动比, 更优波动率)
    try:
        covariance = np.cov(
            high_vol_data['return_vol_ratio'].values,
            high_vol_data['better_volatility'].values
        )[0, 1]
        
        return covariance if not np.isnan(covariance) else None
    except:
        return None


def process_single_stock_climbing_peak(inst: str, df_min: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    单只股票处理（多日期）- 勇攀高峰因子核心计算（严格按研报定义）
    
    研报因子构建步骤：
    1. 仅考虑日内，剔除开盘和收盘部分信息（已在SQL层处理）
    2. 计算每分钟的更优波动率和收益波动比
    3. 找到更优波动率 >= mean + std 的部分，作为异常高波动时段
    4. 计算异常高波动时段的协方差
    5. 每月末计算最近20天的协方差均值和标准差，合成勇攀高峰因子
    
    参数:
        inst: 股票代码
        df_min: 该股票的分钟级数据（可能包含多个交易日）
    
    返回:
        包含每日协方差值的DataFrame
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < 20:
            return None
        
        df_min = df_min.sort_values('date').copy()
        
        # 按日期分组处理
        daily_results = []
        
        for trade_date, day_df in df_min.groupby('trade_date'):
            if len(day_df) < BETTER_VOL_WINDOW + 1:
                continue
            
            # 步骤1-2: 计算更优波动率和收益波动比
            day_df_metrics = calculate_intraday_metrics(day_df, window=BETTER_VOL_WINDOW)
            
            if day_df_metrics is None or len(day_df_metrics) < 10:
                continue
            
            # 步骤3-4: 识别异常高波动时段，计算协方差
            covariance = calculate_daily_covariance(day_df_metrics)
            
            if covariance is None or np.isnan(covariance):
                continue
            
            # 记录当日结果
            daily_results.append({
                'date': pd.to_datetime(trade_date),
                'instrument': inst,
                'daily_covariance': covariance,
                'n_minutes': len(day_df_metrics),
                'vol_mean': day_df_metrics['better_volatility'].mean(),
                'vol_std': day_df_metrics['better_volatility'].std()
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
            executor.submit(process_single_stock_climbing_peak, inst, df): inst 
            for inst, df in stock_data_dict.items()
        }
        
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                print(f"股票 {inst} 计算失败: {e}")
    
    return results


def calculate_daily_factor_optimized(start_date: str, end_date: str, 
                                     batch_size: int = CHUNK_SIZE,
                                     use_parallel: bool = True) -> pd.DataFrame:
    """
    高性能日频因子计算（日协方差）
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
                result = process_single_stock_climbing_peak(inst, df)
                if result is not None and not result.empty:
                    batch_results.append(result)
        
        if batch_results:
            df_batch = pd.concat(batch_results, axis=0, ignore_index=True)
            all_daily_factors.append(df_batch)
            batch_time = (datetime.now() - batch_start).total_seconds()
            print(f"第 {batch_idx} 批完成：{len(df_batch)} 行日协方差数据，耗时 {batch_time:.2f}秒")
    
    if not all_daily_factors:
        return pd.DataFrame()
    
    df_all = pd.concat(all_daily_factors, axis=0, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    return df_all.drop_duplicates(subset=['date', 'instrument'], keep='last').sort_values(['date', 'instrument'])


def calculate_climbing_peak_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算勇攀高峰因子（向量化优化版）
    
    研报步骤5：
    每月末分别计算最近20天的协方差的均值和标准差，
    得到"月均攀登"因子和"月稳攀登"因子，
    并将二者等权合成"勇攀高峰"因子。
    
    因子公式：勇攀高峰因子 = (20日协方差均值 + 20日协方差标准差) / 2
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频协方差数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日协方差数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 向量化滚动计算
    min_periods = min(10, FACTOR_WINDOW)  # 至少10天数据即可计算
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    # 计算20日协方差均值（向量化transform）
    df['covariance_mean'] = df.groupby('instrument')['daily_covariance'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 计算20日协方差标准差（向量化transform）
    df['covariance_std'] = df.groupby('instrument')['daily_covariance'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
    )
    
    # 合成勇攀高峰因子 = (均值 + 标准差) / 2
    df['climbing_peak_factor'] = (df['covariance_mean'] + df['covariance_std']) / 2
    
    # 只保留有效因子值
    df_result = df[df['climbing_peak_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    
    return df_result[['date', 'instrument', 'climbing_peak_factor']]


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
    
    print(f"=== {FACTOR_NAME}（勇攀高峰）因子计算 - 修正版（严格遵循研报定义）===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"更优波动率窗口: {BETTER_VOL_WINDOW}分钟OHLC")
    print()
    print("【研报核心逻辑】")
    print("1. 更优波动率 = (5分钟OHLC共20个价格的std/mean)²")
    print("2. 收益波动比 = 分钟收益率 / 更优波动率")
    print("3. 异常高波动时段：更优波动率 >= mean + std")
    print("4. 协方差 = cov(收益波动比, 更优波动率) 仅在异常高波动时段")
    print("5. 勇攀高峰因子 = (20日协方差均值 + 20日协方差标准差) / 2")
    print()
    
    # 增量计算
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            # 增量起始日期需要往前推足够天数（FACTOR_WINDOW + 缓冲），确保滚动窗口有数据
            buffer_days = SAFETY_BUFFER_DAYS + FACTOR_WINDOW + 10
            effective_start = (pd.to_datetime(last_computed) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"【增量模式】检测到最后计算日期: {last_computed}")
            print(f"【增量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
        else:
            # 全量模式：也需要往前推足够天数
            buffer_days = SAFETY_BUFFER_DAYS + FACTOR_WINDOW + 10
            effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    else:
        buffer_days = SAFETY_BUFFER_DAYS + FACTOR_WINDOW + 10
        effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
        print(f"【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    
    # 步骤1：计算日频协方差
    print("\n【步骤1】计算日频协方差（严格按研报定义）...")
    start_time_calc = datetime.now()
    df_daily = calculate_daily_factor_optimized(effective_start, end_date, 
                                                 batch_size=CHUNK_SIZE,
                                                 use_parallel=use_parallel)
    calc_time = (datetime.now() - start_time_calc).total_seconds()
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频协方差数据，流程终止。")
    else:
        print(f"\n日频协方差计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        print(f"日协方差统计: 均值={df_daily['daily_covariance'].mean():.6f}, 标准差={df_daily['daily_covariance'].std():.6f}")
        
        # 步骤2：计算勇攀高峰因子（20日低频化）
        print("\n【步骤2】计算勇攀高峰因子（20日低频化）...")
        start_time_factor = datetime.now()
        df_factor = calculate_climbing_peak_factor_optimized(df_daily)
        factor_time = (datetime.now() - start_time_factor).total_seconds()
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日协方差数据列: {df_daily.columns.tolist()}")
            print(f"  日协方差数据统计:")
            print(f"    - daily_covariance: 均值={df_daily['daily_covariance'].mean():.6f}, 非空数={df_daily['daily_covariance'].notna().sum()}")
            print(f"  建议检查：")
            print(f"    1. 日期范围是否足够（至少需要{FACTOR_WINDOW}个交易日）")
            print(f"    2. 日协方差数据是否有空值")
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
                
                total_time = (datetime.now() - datetime.fromtimestamp(start_time_total)).total_seconds()
                print(f"\n=== 完成 ===")
                print(f"日协方差计算: {calc_time:.2f}秒")
                print(f"因子合成: {factor_time:.2f}秒")
                print(f"数据写入: {write_time:.2f}秒")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"写入 {written_count} 条数据")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")

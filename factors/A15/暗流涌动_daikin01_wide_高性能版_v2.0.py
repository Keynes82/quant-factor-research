# A15 暗流涌动因子 - 高性能优化版（8核并行+Numba加速）v2.1
# 来源：方正证券-多因子选股系列研究之二十三
#   《个股日内成交量分布特征与日内流动性弹性刻画》（2025-08-27）
# 档案编号：FQ-20260407-A15
# 版本：v2.1 - Numba核心计算加速（基于v2.0）
#
# 【v2.1 优化记录】
#   - 优化1（性能）：核心函数Numba化（@njit cache=True）
#       * 香农熵计算：_shannon_entropy_numba（纯C循环，无Python开销）
#       * 流动性弹性计算：_liquidity_elasticity_numba（纯C循环，无Python开销）
#   - 优化2（兼容）：保留纯Python回退路径（未安装Numba时自动降级）
#   - 优化3（内存）：process_single_stock_a15减少numpy数组重复创建
#
# 【v2.0 重写记录】
#   - 修复1（严重）：核心指标从基尼系数改为香农熵（Shannon Entropy）
#   - 修复2（严重）：新增子因子A（成交量分布熵值），原代码完全缺失
#   - 修复3（严重）：新增子因子B（日内流动性弹性），原代码完全缺失
#   - 修复4（严重）：新增月均+月稳合成结构，原代码只有简单MA
#   - 修复5（严重）：新增截面均值距离化处理，原代码完全缺失
#   - 修复6（严重）：最终因子从单指标改为"暗流涌动"双因子等权合成
#
# 因子构建逻辑（严格按研报原文，共4层）：
#   子因子A：成交量分布熵值
#     1) 日内分钟数据 → 48个5分钟区间
#     2) 区间i相对成交量占比 p(xi) = 区间i成交量 / 全天成交量
#     3) 香农熵 H = -∑ p(xi)·log₂(p(xi))
#     4) 截面均值距离化：|H - 截面均值|
#     5) 低频化：月均(20日MA) + 月稳(20日STD) 等权
#   子因子B：日内流动性弹性
#     1) 剔除开盘(09:30)和收盘(14:57-15:00)，保留09:31-14:56
#     2) 激增时刻：volume_t > 2 × mean(volume_{t-5}..volume_{t-1})
#     3) 价格波动幅度 = (high - low) / open
#     4) 价格敏感系数 = 激增时刻价格波动幅度均值 / 普通时刻价格波动幅度均值
#     5) 弹性系数 = 1 - 价格敏感系数
#     6) 截面均值距离化：|弹性系数 - 截面均值|
#     7) 低频化：月均(20日MA) + 月稳(20日STD) 等权
#   最终因子：暗流涌动 = (子因子A + 子因子B) / 2
#
# 预期表现：
#   子因子A（成交量熵值）：Rank IC = -5.72%
#   子因子B（流动性弹性）：Rank IC = -7.14%
#   暗流涌动（合成）：Rank IC = -7.65%，Rank ICIR = -4.44
#
# 优化点：
#   - 多进程并行计算（8核）
#   - 批量SQL查询（array_agg聚合减少IO）
#   - 向量化香农熵与弹性系数计算
#   - 分层合成：日频原始 → 截面距离化 → 月均+月稳 → 等权合成

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ========== Numba 加速配置 ==========
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    print("Numba 加速已启用")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba 未安装，使用纯 Python 版本（建议安装：pip install numba）")

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A15"                # 本因子编号（暗流涌动）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
MIN_MINUTES_PER_DAY = 30           # 单日最小分钟数（用于子因子2）
N_INTERVALS = 48                   # 日内5分钟区间数（子因子1）
INTERVAL_MINUTES = 5               # 每个区间分钟数
SURGE_MULTIPLIER = 2.0           # 激增倍数（> 2×过去5分钟均值）
EPSILON = 1e-10                    # 防止除零和对数零

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
    
    if 'anliu_yongdong' in df.columns:
        df.rename(columns={'anliu_yongdong': factor_name}, inplace=True)
    
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
    """批量读取分钟级数据（高性能版）- 获取volume/open/high/low序列"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    
    # 获取09:30-15:00全部分钟数据（子因子1需要全天，子因子2在Python中剔除首尾）
    sql = f"""
    SELECT 
        CAST(date AS DATE) as trade_date,
        instrument,
        array_agg(volume ORDER BY date) as volume_series,
        array_agg(open ORDER BY date) as open_series,
        array_agg(high ORDER BY date) as high_series,
        array_agg(low ORDER BY date) as low_series,
        COUNT(*) as n_minutes
    FROM cn_stock_bar1m
    WHERE date >= TIMESTAMP '{start_date} 09:30:00'
      AND date <= TIMESTAMP '{end_date} 15:00:00'
      AND (CAST(date AS TIME) >= '09:30:00' AND CAST(date AS TIME) <= '15:00:00')
      AND instrument IN ('{instrument_list}')
    GROUP BY CAST(date AS DATE), instrument
    HAVING COUNT(*) >= {MIN_MINUTES_PER_DAY}
    ORDER BY trade_date, instrument
    """
    
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['instrument'] = df['instrument'].astype(str)
        df['n_minutes'] = df['n_minutes'].astype(int)
        
        return df
    except Exception as e:
        print(f"SQL查询失败: {e}")
        return pd.DataFrame()


# ========== A15核心：子因子A（成交量分布熵值）- Numba加速版 ==========

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _shannon_entropy_numba(volumes, n_intervals, epsilon):
        """Numba加速版香农熵核心计算"""
        n = len(volumes)
        if n < n_intervals:
            return np.nan
        
        total = 0.0
        for i in range(n):
            total += volumes[i]
        
        if total <= 0:
            return np.nan
        
        interval_size = n // n_intervals
        if interval_size < 1:
            return np.nan
        
        # 区间求和
        interval_vols = np.zeros(n_intervals, dtype=np.float64)
        for i in range(n_intervals):
            start = i * interval_size
            end = n if i == n_intervals - 1 else (i + 1) * interval_size
            s = 0.0
            for j in range(start, end):
                s += volumes[j]
            interval_vols[i] = s
        
        # 香农熵
        entropy = 0.0
        for i in range(n_intervals):
            p = interval_vols[i] / total
            if p > epsilon:
                entropy -= p * np.log2(p)
        
        return entropy


def calculate_shannon_entropy(volume_series, n_intervals=N_INTERVALS):
    """
    计算成交量分布香农熵（Numba加速版）
    
    步骤：
    1) 将分钟序列划分为n_intervals个等长区间
    2) 计算每个区间的成交量总和
    3) p(xi) = 区间i成交量 / 全天成交量
    4) H = -∑ p(xi)·log₂(p(xi))
    
    参数:
        volume_series: 日内成交量序列（分钟级，list或numpy数组）
        n_intervals: 区间数（默认48个5分钟区间）
    
    返回:
        float: 香农熵值，或None（数据不足）
    """
    try:
        # 统一转换为numpy数组
        if isinstance(volume_series, list):
            volumes = np.array(volume_series, dtype=np.float64)
        else:
            volumes = np.asarray(volume_series, dtype=np.float64)
        
        if NUMBA_AVAILABLE:
            result = _shannon_entropy_numba(volumes, n_intervals, EPSILON)
        else:
            # 纯Python回退版
            n = len(volumes)
            if n < n_intervals:
                return None
            
            total_volume = np.sum(volumes)
            if total_volume <= 0:
                return None
            
            interval_size = n // n_intervals
            if interval_size < 1:
                return None
            
            interval_volumes = []
            for i in range(n_intervals):
                start_idx = i * interval_size
                end_idx = n if i == n_intervals - 1 else (i + 1) * interval_size
                interval_vol = np.sum(volumes[start_idx:end_idx])
                interval_volumes.append(interval_vol)
            
            interval_volumes = np.array(interval_volumes, dtype=np.float64)
            p = interval_volumes / total_volume
            p_nonzero = p[p > EPSILON]
            
            if len(p_nonzero) == 0:
                return 0.0
            
            result = -np.sum(p_nonzero * np.log2(p_nonzero))
        
        # 检查是否为有效数值
        if np.isnan(result) or np.isinf(result):
            return None
        return float(result)
        
    except Exception:
        return None


# ========== A15核心：子因子B（日内流动性弹性）- Numba加速版 ==========

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _liquidity_elasticity_numba(opens, highs, lows, volumes, surge_mult, epsilon):
        """Numba加速版流动性弹性核心计算"""
        n = len(volumes)
        if n < 10:
            return np.nan
        
        # 剔除开盘(索引0)和收盘最后4分钟
        start_idx = 1
        end_idx = n - 4 if n - 4 > start_idx + 1 else start_idx + 1
        n_valid = end_idx - start_idx
        
        if n_valid < 6:
            return np.nan
        
        # 计算价格波动幅度 = (high - low) / open
        price_range = np.empty(n_valid, dtype=np.float64)
        for i in range(n_valid):
            o = opens[start_idx + i]
            price_range[i] = (highs[start_idx + i] - lows[start_idx + i]) / (o + epsilon) if o > 0 else 0.0
        
        # 激增时刻识别（从索引5开始）
        surge_sum = 0.0
        surge_count = 0
        normal_sum = 0.0
        normal_count = 0
        
        for i in range(5, n_valid):
            # 过去5分钟均值
            past_5_mean = 0.0
            for j in range(i - 5, i):
                past_5_mean += volumes[start_idx + j]
            past_5_mean /= 5.0
            
            # 当前分钟成交量
            current_vol = volumes[start_idx + i]
            
            if past_5_mean > 0 and current_vol > surge_mult * past_5_mean:
                surge_sum += price_range[i]
                surge_count += 1
            else:
                normal_sum += price_range[i]
                normal_count += 1
        
        if surge_count == 0 or normal_count == 0 or normal_sum <= 0:
            return np.nan
        
        surge_mean = surge_sum / surge_count
        normal_mean = normal_sum / normal_count
        
        if normal_mean <= 0:
            return np.nan
        
        price_sensitivity = surge_mean / normal_mean
        return 1.0 - price_sensitivity


def calculate_liquidity_elasticity(open_series, high_series, low_series, volume_series):
    """
    计算日内流动性弹性系数（Numba加速版）
    
    步骤：
    1) 剔除开盘第1分钟和收盘前4分钟（保留09:31-14:56）
    2) 激增时刻：volume[i] > 2 × mean(volume[i-5:i])
    3) 价格波动幅度 = (high - low) / open
    4) 价格敏感系数 = 激增时刻价格波动幅度均值 / 普通时刻价格波动幅度均值
    5) 弹性系数 = 1 - 价格敏感系数
    
    参数:
        open_series, high_series, low_series, volume_series: 日内分钟序列（list或numpy数组）
    
    返回:
        float: 弹性系数，或None（数据不足/无激增时刻）
    """
    try:
        # 统一转换为numpy数组
        if isinstance(volume_series, list):
            opens = np.array(open_series, dtype=np.float64)
            highs = np.array(high_series, dtype=np.float64)
            lows = np.array(low_series, dtype=np.float64)
            volumes = np.array(volume_series, dtype=np.float64)
        else:
            opens = np.asarray(open_series, dtype=np.float64)
            highs = np.asarray(high_series, dtype=np.float64)
            lows = np.asarray(low_series, dtype=np.float64)
            volumes = np.asarray(volume_series, dtype=np.float64)
        
        if NUMBA_AVAILABLE:
            result = _liquidity_elasticity_numba(opens, highs, lows, volumes, SURGE_MULTIPLIER, EPSILON)
        else:
            # 纯Python回退版
            n = len(volumes)
            if n < 10:
                return None
            
            start_idx = 1
            end_idx = max(start_idx + 1, n - 4)
            n_valid = end_idx - start_idx
            
            if n_valid < 6:
                return None
            
            volumes_valid = volumes[start_idx:end_idx]
            opens_valid = opens[start_idx:end_idx]
            highs_valid = highs[start_idx:end_idx]
            lows_valid = lows[start_idx:end_idx]
            
            price_range = (highs_valid - lows_valid) / (opens_valid + EPSILON)
            
            surge_mask = np.zeros(n_valid, dtype=bool)
            for i in range(5, n_valid):
                past_5_mean = np.mean(volumes_valid[i - 5:i])
                if past_5_mean > 0 and volumes_valid[i] > SURGE_MULTIPLIER * past_5_mean:
                    surge_mask[i] = True
            
            surge_ranges = price_range[surge_mask]
            normal_ranges = price_range[~surge_mask]
            
            if len(surge_ranges) == 0 or len(normal_ranges) == 0:
                return None
            
            surge_range_mean = np.mean(surge_ranges)
            normal_range_mean = np.mean(normal_ranges)
            
            if normal_range_mean <= 0:
                return None
            
            price_sensitivity = surge_range_mean / normal_range_mean
            result = 1.0 - price_sensitivity
        
        # 检查是否为有效数值
        if np.isnan(result) or np.isinf(result):
            return None
        return float(result)
        
    except Exception:
        return None


# ========== 单只股票日频处理（Numba加速版）==========

def process_single_stock_a15(inst: str, df_stock: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    单只股票A15日频计算：子因子A（香农熵）+ 子因子B（流动性弹性）
    
    参数:
        inst: 股票代码
        df_stock: 包含volume_series, open_series, high_series, low_series, n_minutes的DataFrame
    
    返回:
        DataFrame: 每日的子因子A和子因子B原始值
    """
    if df_stock is None or df_stock.empty:
        return None
    
    daily_results = []
    
    for _, row in df_stock.iterrows():
        trade_date = row['trade_date']
        n_minutes = row['n_minutes']
        
        if n_minutes < MIN_MINUTES_PER_DAY:
            continue
        
        # 转换序列（直接使用list或numpy，无需重复创建DataFrame）
        volume_series = row['volume_series']
        open_series = row['open_series']
        high_series = row['high_series']
        low_series = row['low_series']
        
        # 子因子A：成交量分布熵值（Numba加速）
        entropy_raw = calculate_shannon_entropy(volume_series, N_INTERVALS)
        
        # 子因子B：日内流动性弹性（Numba加速）
        elasticity_raw = calculate_liquidity_elasticity(
            open_series, high_series, low_series, volume_series
        )
        
        # 记录结果（允许单个子因子缺失）
        result = {
            'date': trade_date,
            'instrument': inst,
            'n_minutes': n_minutes
        }
        
        if entropy_raw is not None:
            result['entropy_raw'] = entropy_raw
        
        if elasticity_raw is not None:
            result['elasticity_raw'] = elasticity_raw
        
        # 至少有一个子因子有效才记录
        if 'entropy_raw' in result or 'elasticity_raw' in result:
            daily_results.append(result)
    
    if not daily_results:
        return None
    
    return pd.DataFrame(daily_results)


def parallel_process_batch(stock_data_dict: Dict[str, pd.DataFrame], 
                           max_workers: int = MAX_WORKERS) -> List[pd.DataFrame]:
    """并行处理一批股票"""
    if not stock_data_dict:
        return []
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_inst = {
            executor.submit(process_single_stock_a15, inst, df): inst 
            for inst, df in stock_data_dict.items()
        }
        
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
            except Exception:
                pass
    
    return results


def calculate_daily_factor_optimized(start_date: str, end_date: str, 
                                     batch_size: int = CHUNK_SIZE,
                                     use_parallel: bool = True) -> pd.DataFrame:
    """
    高性能日频因子计算（A15暗流涌动版本）
    输出：每只股票每日的原始子因子值（entropy_raw, elasticity_raw）
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
        
        df_batch = fetch_stock_minute_data_batch(batch, start_date, end_date)
        
        if df_batch is None or df_batch.empty:
            continue
        
        stock_groups = {
            inst: group.reset_index(drop=True) 
            for inst, group in df_batch.groupby('instrument')
        }
        
        if use_parallel and len(stock_groups) > 1:
            batch_results = parallel_process_batch(stock_groups, MAX_WORKERS)
        else:
            batch_results = []
            for inst, df in stock_groups.items():
                result = process_single_stock_a15(inst, df)
                if result is not None and not result.empty:
                    batch_results.append(result)
        
        if batch_results:
            df_result = pd.concat(batch_results, axis=0, ignore_index=True)
            all_daily_factors.append(df_result)
            batch_time = (datetime.now() - batch_start).total_seconds()
            print(f"第 {batch_idx} 批完成：{len(df_result)} 行，耗时 {batch_time:.2f}秒")
    
    if not all_daily_factors:
        return pd.DataFrame()
    
    df_all = pd.concat(all_daily_factors, axis=0, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    return df_all.drop_duplicates(subset=['date', 'instrument'], keep='last').sort_values(['date', 'instrument'])


# ========== 截面均值距离化 ==========

def apply_cross_sectional_distance(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    截面均值距离化（严格按研报原文）
    
    对子因子A（entropy_raw）和子因子B（elasticity_raw）分别：
    - 每日截面上，计算所有股票的均值
    - 每只股票的值 = |原始值 - 截面均值|
    
    参数:
        df_daily: DataFrame with [date, instrument, entropy_raw, elasticity_raw]
    
    返回:
        DataFrame: [date, instrument, entropy_dist, elasticity_dist]
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 子因子A：成交量熵值距离化
    if 'entropy_raw' in df.columns:
        daily_entropy_mean = df.groupby('date')['entropy_raw'].transform('mean')
        df['entropy_dist'] = np.abs(df['entropy_raw'] - daily_entropy_mean)
    
    # 子因子B：流动性弹性距离化
    if 'elasticity_raw' in df.columns:
        daily_elasticity_mean = df.groupby('date')['elasticity_raw'].transform('mean')
        df['elasticity_dist'] = np.abs(df['elasticity_raw'] - daily_elasticity_mean)
    
    return df


# ========== 低频化合成（月均+月稳）==========

def calculate_monthly_factor_synthesis(df_dist: pd.DataFrame) -> pd.DataFrame:
    """
    低频化合成：对每个子因子分别计算月均(20日MA) + 月稳(20日STD)，等权合成
    
    子因子A = (月均_A + 月稳_A) / 2
    子因子B = (月均_B + 月稳_B) / 2
    
    参数:
        df_dist: DataFrame with [date, instrument, entropy_dist, elasticity_dist]
    
    返回:
        DataFrame: [date, instrument, sub_a, sub_b, anliu_yongdong]
    """
    if df_dist is None or df_dist.empty:
        return pd.DataFrame()
    
    df = df_dist.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    min_periods = min(10, FACTOR_WINDOW)
    
    # 子因子A：成交量熵值
    if 'entropy_dist' in df.columns:
        df['entropy_ma'] = df.groupby('instrument')['entropy_dist'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
        )
        df['entropy_std'] = df.groupby('instrument')['entropy_dist'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
        )
        # 等权合成（只对有值的项求平均）
        cols_a = ['entropy_ma', 'entropy_std']
        df['sub_factor_a'] = df[cols_a].sum(axis=1) / df[cols_a].notna().sum(axis=1)
        df['sub_factor_a'] = df['sub_factor_a'].replace([np.inf, -np.inf], np.nan)
    
    # 子因子B：流动性弹性
    if 'elasticity_dist' in df.columns:
        df['elasticity_ma'] = df.groupby('instrument')['elasticity_dist'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
        )
        df['elasticity_std'] = df.groupby('instrument')['elasticity_dist'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
        )
        cols_b = ['elasticity_ma', 'elasticity_std']
        df['sub_factor_b'] = df[cols_b].sum(axis=1) / df[cols_b].notna().sum(axis=1)
        df['sub_factor_b'] = df['sub_factor_b'].replace([np.inf, -np.inf], np.nan)
    
    # 最终合成：暗流涌动 = (子因子A + 子因子B) / 2
    final_cols = []
    if 'sub_factor_a' in df.columns:
        final_cols.append('sub_factor_a')
    if 'sub_factor_b' in df.columns:
        final_cols.append('sub_factor_b')
    
    if len(final_cols) > 0:
        df['anliu_yongdong'] = df[final_cols].sum(axis=1) / df[final_cols].notna().sum(axis=1)
        df['anliu_yongdong'] = df['anliu_yongdong'].replace([np.inf, -np.inf], np.nan)
    else:
        df['anliu_yongdong'] = np.nan
    
    # 只保留有最终因子值的数据
    df_result = df[df['anliu_yongdong'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    if len(df_result) > 0:
        print(f"  暗流涌动统计: 均值={df_result['anliu_yongdong'].mean():.6f}, 标准差={df_result['anliu_yongdong'].std():.6f}")
        if 'sub_factor_a' in df_result.columns:
            print(f"  子因子A(成交量熵值): 均值={df_result['sub_factor_a'].mean():.6f}")
        if 'sub_factor_b' in df_result.columns:
            print(f"  子因子B(流动性弹性): 均值={df_result['sub_factor_b'].mean():.6f}")
    
    return df_result[['date', 'instrument', 'sub_factor_a', 'sub_factor_b', 'anliu_yongdong']]


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
    
    print(f"=== {FACTOR_NAME}（暗流涌动）因子计算 - v2.1 Numba加速版 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日（月均+月稳合成）")
    print(f"【v2.1优化】Numba核心计算加速（香农熵+流动性弹性）")
    print(f"【v2.0重写】香农熵 + 流动性弹性 + 截面距离化 + 双层等权合成")
    print(f"子因子A: 48区间成交量分布熵值")
    print(f"子因子B: 激增时刻价格敏感系数 → 弹性系数")
    
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
    
    # 步骤1：计算日频原始子因子值
    print("\n【步骤1】计算日频原始子因子（并行优化）...")
    start_time_calc = time.time()
    df_daily_raw = calculate_daily_factor_optimized(effective_start, end_date, 
                                                     batch_size=CHUNK_SIZE,
                                                     use_parallel=use_parallel)
    calc_time = time.time() - start_time_calc
    
    if df_daily_raw is None or df_daily_raw.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频计算完成: {len(df_daily_raw)} 条，耗时 {calc_time:.2f}秒")
        print(f"  子因子A(熵值)非空: {df_daily_raw['entropy_raw'].notna().sum()}")
        print(f"  子因子B(弹性)非空: {df_daily_raw['elasticity_raw'].notna().sum()}")
        
        # 步骤2：截面均值距离化
        print("\n【步骤2】截面均值距离化...")
        start_time_dist = time.time()
        df_dist = apply_cross_sectional_distance(df_daily_raw)
        dist_time = time.time() - start_time_dist
        print(f"截面距离化完成: 耗时 {dist_time:.2f}秒")
        
        # 步骤3：低频化合成（月均+月稳）
        print("\n【步骤3】低频化合成（月均+月稳 → 子因子A/B → 暗流涌动）...")
        start_time_factor = time.time()
        df_factor = calculate_monthly_factor_synthesis(df_dist)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日频数据列: {df_dist.columns.tolist()}")
            print(f"  建议检查：")
            print(f"    1. 日期范围是否足够（至少需要{FACTOR_WINDOW}个交易日）")
            print(f"    2. 日频数据是否有空值")
        else:
            print(f"因子合成完成: {len(df_factor)} 条，耗时 {factor_time:.2f}秒")
            
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
                print(f"截面距离化: {dist_time:.2f}秒")
                print(f"低频合成: {factor_time:.2f}秒")
                print(f"数据写入: {write_time:.2f}秒")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"写入 {written_count} 条数据")
                print(f"\n预期表现:")
                print(f"  子因子A(成交量熵值): Rank IC ~ -5.72%")
                print(f"  子因子B(流动性弹性): Rank IC ~ -7.14%")
                print(f"  暗流涌动(合成): Rank IC ~ -7.65%, Rank ICIR ~ -4.44")
                print(f"  多空年化收益率: ~29.17%, 月度胜率: ~88.98%")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily_raw) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print(f"日期过滤后无数据，尝试扩大计算范围。")

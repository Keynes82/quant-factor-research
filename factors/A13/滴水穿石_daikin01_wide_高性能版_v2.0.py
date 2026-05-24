# A13 滴水穿石因子 - 高性能优化版（8核并行）v2.0
# 来源：方正证券-多因子选股系列研究之二十四《滴水穿石》（2025-12-16）
# 档案编号：FQ-20260407-A13
# 版本：v2.0 - 严格按研报原文重写，修复核心逻辑偏差
#
# 【v2.0 重写记录】
#   - 修复1（严重）：新增 IQR 限幅去脉冲处理（研报步骤2，原代码完全缺失）
#   - 修复2（严重）：新增 Hann 窗函数（研报步骤3，原代码完全缺失）
#   - 修复3（严重）：改为提取 2-5分钟频带能量占比（研报步骤5，原代码提取的是"主频率+能量集中度"）
#   - 修复4（严重）：因子公式改为 band_power / (total_power + ε)（研报步骤6，原代码是 dominant_ratio × concentration）
#
# 因子构建逻辑（严格按研报原文，共6步）：
#   1. 数据清理：剔除开盘1分钟+收盘前3分钟（保留 09:31-11:30, 13:00-14:56）
#   2. 去脉冲处理：IQR限幅 clip(x, median±3·IQR)
#   3. 去均值 + Hann窗：xw(t) = (x - mean(x)) · w(t)
#   4. rFFT + 功率谱：P(f) = |rfft(xw)|²
#   5. 提取2-5分钟频带能量：band_mask = {f | period(f) ∈ [2,5]分钟}
#   6. 滴水穿石因子 = band_power / (total_power + ε)
#   7. 月因子 = 过去20个交易日日因子的均值
#
# 优化点：
#   - 多进程并行计算（8核）
#   - 批量SQL查询（array_agg聚合减少IO）
#   - numpy.rfft高效频谱计算
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
FACTOR_NAME = "A13"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
MIN_MINUTES_FOR_FFT = 30           # FFT计算所需最小分钟数

# ========== 性能配置 ==========
MAX_WORKERS = 8                    # 并行进程数
CHUNK_SIZE = 1000                  # 批量处理大小

# ========== FFT参数配置 ==========
BAND_PERIOD_MIN = 2.0              # 目标频带周期下限（分钟）
BAND_PERIOD_MAX = 5.0              # 目标频带周期上限（分钟）
IQR_MULTIPLIER = 3.0               # IQR限幅倍数
EPSILON = 1e-10                    # 防止除零


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
    
    if 'drip_stone_factor' in df.columns:
        df.rename(columns={'drip_stone_factor': factor_name}, inplace=True)
    
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
    """批量读取分钟级数据（高性能版）- 使用array_agg聚合减少数据量"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    
    # 研报要求：剔除开盘1分钟+收盘前3分钟 → 保留 09:31-11:30, 13:00-14:56
    sql = f"""
    SELECT 
        CAST(date AS DATE) as trade_date,
        instrument,
        array_agg(volume ORDER BY date) as volume_series,
        COUNT(*) as n_minutes
    FROM cn_stock_bar1m
    WHERE date >= TIMESTAMP '{start_date} 09:30:00'
      AND date <= TIMESTAMP '{end_date} 15:00:00'
      AND (
          (CAST(date AS TIME) >= '09:31:00' AND CAST(date AS TIME) <= '11:30:00')
          OR 
          (CAST(date AS TIME) >= '13:00:00' AND CAST(date AS TIME) <= '14:56:00')
      )
      AND instrument IN ('{instrument_list}')
    GROUP BY CAST(date AS DATE), instrument
    HAVING COUNT(*) >= {MIN_MINUTES_FOR_FFT}
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


# ========== A13因子核心计算函数（严格按研报原文 v2.0重写） ==========

def calculate_drip_stone_features(volume_series: np.ndarray) -> Optional[float]:
    """
    严格按研报原文计算滴水穿石因子（6步构建）
    
    研报步骤：
    1) 数据清理（SQL层已完成）
    2) IQR限幅去脉冲：clip(x, median±3·IQR)
    3) 去均值 + Hann窗
    4) rFFT + 功率谱
    5) 提取2-5分钟频带能量
    6) 因子 = band_power / (total_power + ε)
    
    参数:
        volume_series: 日内成交量序列（已剔除开盘1分钟+收盘前3分钟）
    
    返回:
        float: 滴水穿石日频因子值，或None（数据不足/计算失败）
    """
    try:
        if len(volume_series) < MIN_MINUTES_FOR_FFT:
            return None
        
        volume_series = np.array(volume_series, dtype=np.float64)
        
        # === 步骤2：去脉冲处理（IQR限幅）===
        # 研报原文：x(t) = clip(x(t), lower=median(x)-3·IQR, upper=median(x)+3·IQR)
        q25, median, q75 = np.percentile(volume_series, [25, 50, 75])
        iqr = q75 - q25
        lower = median - IQR_MULTIPLIER * iqr
        upper = median + IQR_MULTIPLIER * iqr
        volume_clipped = np.clip(volume_series, lower, upper)
        
        # === 步骤3：去均值 + Hann窗 ===
        # 研报原文：x̃(t) = x(t) - mean(x);  xw(t) = x̃(t) · w(t)
        volume_centered = volume_clipped - np.mean(volume_clipped)
        n = len(volume_centered)
        hann_window = np.hanning(n)
        volume_windowed = volume_centered * hann_window
        
        # === 步骤4：rFFT + 功率谱 ===
        # 研报原文：FFTc(f) = rfft(xw(t));  P(f) = |FFTc(f)|²
        fft_result = np.fft.rfft(volume_windowed)
        energy_spectrum = np.abs(fft_result) ** 2
        
        if np.sum(energy_spectrum) == 0:
            return None
        
        # === 步骤5：提取2-5分钟频带能量 ===
        # 研报原文：period(f) = 1/freq(f);  band_mask = {f | period(f) ∈ [2,5]分钟}
        freqs = np.fft.rfftfreq(n, d=1.0)  # d=1分钟，采样间隔1分钟
        
        # 排除DC分量（freq=0），计算各频率对应的周期
        non_dc_mask = freqs > 0
        if not np.any(non_dc_mask):
            return None
        
        # period = 1/freq，排除DC
        periods = np.zeros_like(freqs)
        periods[non_dc_mask] = 1.0 / freqs[non_dc_mask]
        
        # 2-5分钟频带mask
        band_mask = non_dc_mask & (periods >= BAND_PERIOD_MIN) & (periods <= BAND_PERIOD_MAX)
        
        if not np.any(band_mask):
            return None
        
        # band_power = Σ_{f ∈ band_mask} P(f)
        band_power = np.sum(energy_spectrum[band_mask])
        # total_power = Σ_{f ≠ 0} P(f)（排除DC）
        total_power = np.sum(energy_spectrum[non_dc_mask])
        
        if total_power <= 0:
            return None
        
        # === 步骤6：滴水穿石因子 ===
        # 研报原文：滴水穿石因子 = band_power / (total_power + ε)
        drip_stone_factor = band_power / (total_power + EPSILON)
        
        return float(drip_stone_factor)
        
    except Exception:
        return None


def process_single_stock_drip_stone(inst: str, df_stock: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    单只股票滴水穿石因子计算（v2.0重写版）
    
    参数:
        inst: 股票代码
        df_stock: 该股票的数据DataFrame，包含volume_series列
    
    返回:
        DataFrame: 包含每日滴水穿石因子值
    """
    try:
        if df_stock is None or df_stock.empty:
            return None
        
        daily_results = []
        
        for _, row in df_stock.iterrows():
            trade_date = row['trade_date']
            volume_series = row['volume_series']
            n_minutes = row['n_minutes']
            
            # 检查数据完整性
            if n_minutes < MIN_MINUTES_FOR_FFT:
                continue
            
            # 转换volume_series为numpy数组
            if isinstance(volume_series, list):
                volume_array = np.array(volume_series, dtype=np.float64)
            elif isinstance(volume_series, np.ndarray):
                volume_array = volume_series.astype(np.float64)
            else:
                continue
            
            # 计算滴水穿石因子（严格按研报6步）
            daily_factor = calculate_drip_stone_features(volume_array)
            
            if daily_factor is not None and not np.isnan(daily_factor) and not np.isinf(daily_factor):
                daily_results.append({
                    'date': trade_date,
                    'instrument': inst,
                    'daily_drip_stone_factor': daily_factor,
                    'n_minutes': n_minutes
                })
        
        if not daily_results:
            return None
        
        return pd.DataFrame(daily_results)
        
    except Exception:
        return None


def parallel_process_batch(stock_data_dict: Dict[str, pd.DataFrame], 
                            max_workers: int = MAX_WORKERS) -> List[pd.DataFrame]:
    """并行处理一批股票的滴水穿石因子计算"""
    if not stock_data_dict:
        return []
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_inst = {
            executor.submit(process_single_stock_drip_stone, inst, df): inst 
            for inst, df in stock_data_dict.items()
        }
        
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
            except Exception:
                pass  # 静默处理单个股票错误
    
    return results


def calculate_daily_factor_optimized(start_date: str, end_date: str, 
                                     batch_size: int = CHUNK_SIZE,
                                     use_parallel: bool = True) -> pd.DataFrame:
    """
    高性能日频因子计算（滴水穿石版本）
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
        
        # 批量获取数据（SQL层已聚合）
        df_batch = fetch_stock_minute_data_batch(batch, start_date, end_date)
        
        if df_batch is None or df_batch.empty:
            continue
        
        # 分组为每只股票的数据
        stock_groups = {
            inst: group.reset_index(drop=True) 
            for inst, group in df_batch.groupby('instrument')
        }
        
        # 并行或串行处理
        if use_parallel and len(stock_groups) > 1:
            batch_results = parallel_process_batch(stock_groups, MAX_WORKERS)
        else:
            batch_results = []
            for inst, df in stock_groups.items():
                result = process_single_stock_drip_stone(inst, df)
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


def calculate_drip_stone_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算滴水穿石月频因子（向量化滚动优化版）
    
    月因子 = 过去20个交易日日因子的均值
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只，日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 向量化滚动计算月因子（20日平均）
    min_periods = min(10, FACTOR_WINDOW)  # 至少10天数据即可计算
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    df['drip_stone_factor'] = df.groupby('instrument')['daily_drip_stone_factor'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 只保留有效因子值
    df_result = df[df['drip_stone_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    
    return df_result[['date', 'instrument', 'drip_stone_factor']]


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
    
    print(f"=== {FACTOR_NAME}（滴水穿石）因子计算 - v2.0严格按研报重写 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日, FFT最小样本: {MIN_MINUTES_FOR_FFT}分钟")
    print(f"目标频带: {BAND_PERIOD_MIN}-{BAND_PERIOD_MAX}分钟周期")
    print(f"【v2.0重写】IQR去脉冲 + Hann窗 + 2-5分钟频带能量占比")
    
    # 增量计算
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            # 增量起始日期需要往前推足够天数（FACTOR_WINDOW + 缓冲），确保滚动窗口有数据
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
    
    # 步骤1：计算日频滴水穿石因子
    print("\n【步骤1】计算日频滴水穿石因子（并行优化）...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor_optimized(effective_start, end_date, 
                                                 batch_size=CHUNK_SIZE,
                                                 use_parallel=use_parallel)
    calc_time = time.time() - start_time_calc
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        
        # 步骤2：计算滴水穿石月频因子（20日滚动平均）
        print("\n【步骤2】计算滴水穿石月频因子（向量化滚动平均）...")
        start_time_factor = time.time()
        df_factor = calculate_drip_stone_factor_optimized(df_daily)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日频数据列: {df_daily.columns.tolist()}")
            print(f"  日频数据统计:")
            print(f"    - daily_drip_stone_factor: 均值={df_daily['daily_drip_stone_factor'].mean():.6f}, 非空数={df_daily['daily_drip_stone_factor'].notna().sum()}")
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
                print(f"\n预期表现: Rank IC ~ 8.69%")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print(f"日期过滤后无数据，尝试扩大计算范围。")

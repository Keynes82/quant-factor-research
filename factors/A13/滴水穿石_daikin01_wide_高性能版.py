# A13 滴水穿石因子 - 高性能优化版（8核并行）
# 来源：基于FFT频谱分析的成交量特征因子
# 档案编号：FQ-20260407-A13
# 版本：v1.0 - 基于高性能模板（多进程并行优化）
#
# 因子构建逻辑：
#   1. 获取日内分钟级成交量数据
#   2. 对成交量序列进行FFT（快速傅里叶变换）
#   3. 提取频谱特征（主频率成分能量）
#   4. 日因子 = 主频率能量占比 × 能量集中度
#   5. 月因子 = 过去20日平均值
#
# 优化点：
#   - 多进程并行计算（8核）
#   - 批量SQL查询（减少IO）
#   - numpy.fft高效频谱计算
#   - 动态增量缓冲

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
FFT_NORM = True                    # FFT归一化
ENERGY_TOP_K = 3                   # 能量集中度计算时取前K个频率成分


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
    
    if 'fft_factor' in df.columns:
        df.rename(columns={'fft_factor': factor_name}, inplace=True)
    
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
    
    # 使用array_agg聚合分钟数据，按交易日返回，减少网络传输
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


# ========== FFT频谱计算函数 ==========

def calculate_fft_features(volume_series: np.ndarray, top_k: int = ENERGY_TOP_K) -> Optional[Dict]:
    """
    计算FFT频谱特征
    
    参数:
        volume_series: 成交量序列
        top_k: 能量集中度计算时取前K个频率成分
    
    返回:
        dict: 包含主频率能量占比和能量集中度的字典
    """
    try:
        if len(volume_series) < MIN_MINUTES_FOR_FFT:
            return None
        
        # 数据预处理：去除趋势（零均值化）
        volume_centered = volume_series - np.mean(volume_series)
        
        # FFT计算（使用numpy.fft优化）
        fft_result = np.fft.fft(volume_centered)
        
        # 计算频谱能量（振幅平方）
        # 只取前半部分频谱（对称性）
        n = len(volume_centered)
        energy_spectrum = np.abs(fft_result[:n//2]) ** 2
        
        if np.sum(energy_spectrum) == 0:
            return None
        
        # 计算能量占比
        total_energy = np.sum(energy_spectrum)
        
        # 找到主频率（能量最大的频率成分，排除DC分量）
        # DC分量（索引0）是均值信息，从索引1开始
        if len(energy_spectrum) > 1:
            dominant_idx = np.argmax(energy_spectrum[1:]) + 1
            dominant_energy = energy_spectrum[dominant_idx]
        else:
            return None
        
        # 主频率能量占比
        dominant_energy_ratio = dominant_energy / total_energy
        
        # 能量集中度：前top_k个频率成分的能量占比
        top_k_indices = np.argsort(energy_spectrum[1:])[-top_k:] + 1
        top_k_energy = np.sum(energy_spectrum[top_k_indices])
        energy_concentration = top_k_energy / total_energy
        
        return {
            'dominant_energy_ratio': dominant_energy_ratio,
            'energy_concentration': energy_concentration,
            'total_energy': total_energy,
            'n_frequencies': len(energy_spectrum)
        }
    except Exception as e:
        return None


def process_single_stock_fft(inst: str, df_stock: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    单只股票FFT特征计算
    
    参数:
        inst: 股票代码
        df_stock: 该股票的数据DataFrame，包含volume_series列
    
    返回:
        DataFrame: 包含每日FFT因子值
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
            
            # 计算FFT特征
            fft_features = calculate_fft_features(volume_array)
            
            if fft_features is not None:
                # 计算日因子值 = 主频率能量占比 × 能量集中度
                daily_factor = fft_features['dominant_energy_ratio'] * fft_features['energy_concentration']
                
                daily_results.append({
                    'date': trade_date,
                    'instrument': inst,
                    'daily_fft_factor': daily_factor,
                    'dominant_energy_ratio': fft_features['dominant_energy_ratio'],
                    'energy_concentration': fft_features['energy_concentration'],
                    'total_energy': fft_features['total_energy'],
                    'n_minutes': n_minutes
                })
        
        if not daily_results:
            return None
        
        return pd.DataFrame(daily_results)
        
    except Exception as e:
        return None


def parallel_process_batch_fft(stock_data_dict: Dict[str, pd.DataFrame], 
                                max_workers: int = MAX_WORKERS) -> List[pd.DataFrame]:
    """并行处理一批股票的FFT计算"""
    if not stock_data_dict:
        return []
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_inst = {
            executor.submit(process_single_stock_fft, inst, df): inst 
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
    高性能日频因子计算（FFT版本）
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
            batch_results = parallel_process_batch_fft(stock_groups, MAX_WORKERS)
        else:
            batch_results = []
            for inst, df in stock_groups.items():
                result = process_single_stock_fft(inst, df)
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


def calculate_monthly_fft_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算滴水穿石因子（向量化滚动优化版）
    
    月因子 = 过去20日FFT因子的平均值
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
    
    df['fft_factor'] = df.groupby('instrument')['daily_fft_factor'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 只保留有效因子值
    df_result = df[df['fft_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    
    return df_result[['date', 'instrument', 'fft_factor']]


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
    
    print(f"=== {FACTOR_NAME}（滴水穿石）因子计算 - 高性能优化版 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日, FFT最小样本: {MIN_MINUTES_FOR_FFT}分钟")
    
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
    
    # 步骤1：计算日频FFT因子
    print("\n【步骤1】计算日频FFT因子（并行优化）...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor_optimized(effective_start, end_date, 
                                                 batch_size=CHUNK_SIZE,
                                                 use_parallel=use_parallel)
    calc_time = time.time() - start_time_calc
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        
        # 步骤2：计算滴水穿石因子（20日滚动平均）
        print("\n【步骤2】计算滴水穿石因子（向量化滚动平均）...")
        start_time_factor = time.time()
        df_factor = calculate_monthly_fft_factor_optimized(df_daily)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日频数据列: {df_daily.columns.tolist()}")
            print(f"  日频数据统计:")
            print(f"    - daily_fft_factor: 均值={df_daily['daily_fft_factor'].mean():.6f}, 非空数={df_daily['daily_fft_factor'].notna().sum()}")
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
                print(f"\n因子逻辑: FFT主频率能量占比 × 能量集中度，再取20日平均")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")

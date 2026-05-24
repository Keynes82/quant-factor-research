# A12 一视同仁因子 - 高性能优化版（8核并行）
# 来源：方正证券-多因子选股系列研究之十八《一视同仁》（2024-05-23）
# 档案编号：FQ-20240320-001
# 版本：v2.0 - 基于高性能模板（多进程并行优化）
#
# 因子构建逻辑：
#   1. 对成交量做boxcox变换使其正态化
#   2. 计算每分钟成交量变化量
#   3. 定义"正态激增时刻"（变化量>mean+std）和"正态骤降时刻"（变化量<mean-std）
#   4. 计算波动公平因子：激增和骤降时刻波动率差异的绝对值 × 日内收益率
#   5. 计算收益公平因子：激增和骤降时刻收益率差异的绝对值 × 日内收益率
#   6. 合成"一视同仁"因子：波动公平 + 收益公平（等权合成）
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A12"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口
BRILLIANT_MINUTES = 5              # 耀眼/黯淡N分钟

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
    
    if 'fairness_factor' in df.columns:
        df.rename(columns={'fairness_factor': factor_name}, inplace=True)
    
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

def apply_boxcox_transform(volume_series: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    对成交量序列进行boxcox变换，使其正态化
    要求所有值为正数
    """
    # 过滤掉0值和负数
    positive_mask = volume_series > 0
    if positive_mask.sum() < 10:  # 需要足够的数据点
        return None, None
    
    positive_volumes = volume_series[positive_mask]
    
    try:
        # boxcox变换
        transformed, lambda_param = stats.boxcox(positive_volumes)
        return transformed, lambda_param
    except Exception:
        return None, None


def process_single_stock_optimized(inst: str, df_min: pd.DataFrame, 
                                   brilliant_minutes: int = BRILLIANT_MINUTES) -> Optional[pd.DataFrame]:
    """
    单只股票处理（向量化优化）- 一视同仁因子核心逻辑
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < 30:
            return None
        
        df_min = df_min.sort_values('date').copy()
        
        # 修复：按日期分组处理，确保每个交易日都有数据
        daily_results = []
        
        for trade_date, day_df in df_min.groupby('trade_date'):
            if len(day_df) < 30:
                continue
            
            day_df = day_df.sort_values('date').reset_index(drop=True)
            
            # 步骤1：对成交量进行boxcox变换
            transformed_volumes, lambda_param = apply_boxcox_transform(day_df['volume'].values)
            
            if transformed_volumes is None:
                continue
            
            # 步骤2：计算成交量变化量
            volume_changes = np.diff(transformed_volumes)
            
            if len(volume_changes) < 10:
                continue
            
            # 步骤3：计算变化量的均值和标准差
            mean_change = np.mean(volume_changes)
            std_change = np.std(volume_changes)
            
            if std_change == 0 or np.isnan(mean_change) or np.isnan(std_change):
                continue
            
            # 步骤4：识别正态激增时刻和正态骤降时刻
            surge_threshold = mean_change + std_change
            drop_threshold = mean_change - std_change
            
            surge_indices = np.where(volume_changes > surge_threshold)[0]
            drop_indices = np.where(volume_changes < drop_threshold)[0]
            
            if len(surge_indices) == 0 or len(drop_indices) == 0:
                continue
            
            # 计算每分钟的收益率
            day_df['return'] = day_df['close'].pct_change()
            
            # 步骤5：识别"正态耀眼五分钟"和"正态黯淡五分钟"
            brilliant_periods = set()
            dim_periods = set()
            
            # 激增时刻及其后4分钟
            for idx in surge_indices:
                for i in range(brilliant_minutes):
                    if idx + i < len(day_df):
                        brilliant_periods.add(idx + i)
            
            # 骤降时刻及其后4分钟
            for idx in drop_indices:
                for i in range(brilliant_minutes):
                    if idx + i < len(day_df):
                        dim_periods.add(idx + i)
            
            if len(brilliant_periods) == 0 or len(dim_periods) == 0:
                continue
            
            # 计算两种期间的波动率（使用收益率标准差）
            brilliant_vol = day_df.iloc[list(brilliant_periods)]['return'].std()
            dim_vol = day_df.iloc[list(dim_periods)]['return'].std()
            
            if pd.isna(brilliant_vol) or pd.isna(dim_vol):
                continue
            
            # 波动公平度
            volatility_fairness = abs(brilliant_vol - dim_vol)
            
            # 激增时刻和骤降时刻的收益率
            # 注意：surge_indices/drop_indices是基于volume_changes的索引，对应day_df的1:-1
            surge_returns = day_df.iloc[surge_indices + 1]['return']
            drop_returns = day_df.iloc[drop_indices + 1]['return']
            
            if len(surge_returns) == 0 or len(drop_returns) == 0:
                continue
            
            surge_mean_return = surge_returns.mean()
            drop_mean_return = drop_returns.mean()
            
            if pd.isna(surge_mean_return) or pd.isna(drop_mean_return):
                continue
            
            # 收益公平度
            return_fairness = abs(surge_mean_return - drop_mean_return)
            
            # 日内收益率
            daily_return = (day_df['close'].iloc[-1] - day_df['close'].iloc[0]) / day_df['close'].iloc[0]
            
            if pd.isna(daily_return):
                continue
            
            # 步骤6：计算波动公平因子和收益公平因子
            volatility_fair_return = daily_return * volatility_fairness
            return_fair_return = daily_return * return_fairness
            
            # 步骤7：合成一视同仁因子（等权）
            combined_factor = volatility_fair_return + return_fair_return
            
            if not np.isnan(combined_factor) and not np.isinf(combined_factor):
                daily_results.append({
                    'date': pd.to_datetime(trade_date),
                    'instrument': inst,
                    'volatility_fair': volatility_fair_return,
                    'return_fair': return_fair_return,
                    'fairness_factor': combined_factor
                })
        
        if not daily_results:
            return None
        
        return pd.DataFrame(daily_results)
        
    except Exception:
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
            except Exception:
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
                result = process_single_stock_optimized(inst, df)
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


def calculate_fairness_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算一视同仁因子（向量化优化版）
    
    修复：min_periods改为10（允许部分数据计算）
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只，日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 向量化滚动计算 - 修复：min_periods改为10，允许部分数据计算
    min_periods = min(10, FACTOR_WINDOW)  # 至少10天数据即可计算
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    df['fairness_factor'] = df.groupby('instrument')['fairness_factor'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 只保留有效因子值
    df_result = df[df['fairness_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    
    return df_result[['date', 'instrument', 'fairness_factor']]


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
    
    print(f"=== {FACTOR_NAME}（一视同仁）因子计算 - 高性能优化版 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日, 耀眼/黯淡分钟: {BRILLIANT_MINUTES}")
    
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
        
        # 步骤2：计算一视同仁因子（滚动平滑）
        print("\n【步骤2】计算一视同仁因子（向量化滚动）...")
        start_time_factor = time.time()
        df_factor = calculate_fairness_factor_optimized(df_daily)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日频数据列: {df_daily.columns.tolist()}")
            print(f"  日频数据统计:")
            print(f"    - volatility_fair: 均值={df_daily['volatility_fair'].mean():.6f}, 非空数={df_daily['volatility_fair'].notna().sum()}")
            print(f"    - return_fair: 均值={df_daily['return_fair'].mean():.6f}, 非空数={df_daily['return_fair'].notna().sum()}")
            print(f"    - fairness_factor: 均值={df_daily['fairness_factor'].mean():.6f}, 非空数={df_daily['fairness_factor'].notna().sum()}")
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
                print(f"\n预期表现: Rank IC ~ -7.39%")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")

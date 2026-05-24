# A10 待著而救因子 - v2.2 BUG修复版
# 来源：量价关系因子 - 大单跟随效应
# 档案编号：FQ-20260407-A10
# 版本：v2.2 - BUG修复版（修复优势时刻筛选逻辑：应与前一个海量时刻比较，而非前一个优势时刻）
#
# 【v2.2修复内容】
#   修复优势时刻筛选逻辑：
#     原代码：当前海量时刻与"最后一个优势时刻"比较间隔
#     修复后：当前海量时刻与"前一个海量时刻"比较间隔（无论是否被保留）
#     原因：研报原文"相邻的两个海量时刻间隔"指的是所有海量时刻，不是仅保留的优势时刻
#
# 【v2.1修正内容】
#   1. 跟随系数计算口径修正：amount（成交额）→ volume（成交量），与研报原文"成交量"对齐
#   2. 跟随窗口修正：不包含优势时刻本身，仅取"之后"的5分钟
#
# 因子构建逻辑（严格按研报原文）：
#   1. 数据预处理：剔除每天9:45之前的数据
#   2. 识别"海量时刻"：当日成交量最大的10个分钟时刻
#   3. 筛选"优势时刻"：相邻两个海量时刻间隔>5分钟保留，否则剔除后者
#   4. 计算"跟随系数"：跟随时刻成交量 / 对应优势时刻成交量
#   5. 日跟随系数：日内所有跟随系数的均值
#   6. 月均待著而救：过去20日日跟随系数的均值
#   7. 月稳待著而救：过去20日日跟随系数的标准差
#   8. 待著而救因子 = 月均 + 月稳（等权合成）
#
# 优化点：
#   - 多进程并行计算（8核）
#   - 批量SQL查询（减少IO）
#   - NumPy向量化操作

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
FACTOR_NAME = "A10"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（月因子=20日平均）

# ========== A10因子特定参数 ==========
TOP_N_MASSIVE = 10                 # 海量时刻数量（研报原文：10个）
MIN_INTERVAL_MINUTES = 5           # 最小间隔分钟数，避免重复计算
FOLLOW_WINDOW = 5                  # 跟随观察窗口（分钟）

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
    
    if 'final_factor' in df.columns:
        df.rename(columns={'final_factor': factor_name}, inplace=True)
    
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
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) >= 930
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) <= 1500
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


# ========== A10因子核心计算函数（严格按研报原文修正） ==========

def filter_by_market_open(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤1：数据预处理 - 剔除每天9:45之前的数据
    研报原文："由于每日开盘的前15分钟时间内股票交易普遍较为活跃，
             我们对每天9:45分之前的数据进行剔除"
    """
    df['time_str'] = df['date'].dt.strftime('%H:%M')
    df_filtered = df[df['time_str'] >= '09:45'].copy()
    df_filtered = df_filtered.drop(columns=['time_str'])
    return df_filtered


def identify_massive_moments_topn(df: pd.DataFrame, top_n: int = TOP_N_MASSIVE) -> pd.DataFrame:
    """
    步骤2：识别"海量时刻" - 当日成交量最大的10个分钟时刻
    研报原文："找到当日成交量最大的十个分钟时刻，将其统称为'海量时刻'"
    """
    if len(df) < top_n:
        df['is_massive'] = False
        return df
    
    # 按成交量排序，取前top_n个
    top_indices = df['volume'].nlargest(top_n).index
    df['is_massive'] = False
    df.loc[top_indices, 'is_massive'] = True
    
    return df


def filter_advantage_moments(df: pd.DataFrame, 
                             min_interval: int = MIN_INTERVAL_MINUTES) -> pd.DataFrame:
    """
    步骤3：筛选"优势时刻"
    研报原文："从时间上最靠前的'海量时刻'开始，
             如果相邻的两个'海量时刻'间隔超过5分钟，保留；
             如果间隔小于5分钟，剔除后者"
    
    【v2.2修复】当前海量时刻与前一个"海量时刻"比较间隔（无论是否被保留），
    而不是与前一个"优势时刻"比较。研报说的是"相邻的两个海量时刻"，
    不是"相邻的两个优势时刻"。
    """
    massive_df = df[df['is_massive']].copy().sort_values('date')
    
    if len(massive_df) == 0:
        df['is_advantage'] = False
        return df
    
    # 获取海量时刻的索引和时间
    massive_indices = massive_df.index.tolist()
    massive_times = massive_df['date'].tolist()
    
    # 从时间最靠前的开始筛选
    valid_indices = [massive_indices[0]]  # 第一个总是有效的
    
    for i in range(1, len(massive_indices)):
        current_time = massive_times[i]
        # 【v2.2修复】与前一个海量时刻比较（不是前一个优势时刻）
        prev_time = massive_times[i - 1]
        
        # 计算时间差（分钟）
        time_diff = (current_time - prev_time).total_seconds() / 60
        
        if time_diff >= min_interval:
            valid_indices.append(massive_indices[i])
    
    # 标记优势时刻
    df['is_advantage'] = False
    df.loc[valid_indices, 'is_advantage'] = True
    
    return df


def calculate_follow_coefficients(df: pd.DataFrame, 
                                  follow_window: int = FOLLOW_WINDOW) -> List[float]:
    """
    步骤4：计算"跟随系数"
    研报原文："计算每个'跟随时刻'的成交量总和，
             并除以对应的'优势时刻'的成交量，得到'跟随系数'"
    
    【v2.1修正】
      1. 口径修正：amount（成交额）→ volume（成交量），与研报原文对齐
      2. 窗口修正：不包含优势时刻本身，仅取"之后"的5分钟
    """
    advantage_indices = np.where(df['is_advantage'].values)[0]
    
    if len(advantage_indices) == 0:
        return []
    
    follow_coefficients = []
    n_rows = len(df)
    
    for idx in advantage_indices:
        # 优势时刻成交量（分母）
        advantage_volume = df.iloc[idx]['volume']
        
        if advantage_volume <= 0 or pd.isna(advantage_volume):
            continue
        
        # 【v2.1修正】跟随时刻：优势时刻"之后"的5分钟（不包含优势时刻本身）
        start_idx = idx + 1
        if start_idx >= n_rows:
            continue
        end_idx = min(start_idx + follow_window, n_rows)
        follow_volume = df.iloc[start_idx:end_idx]['volume'].sum()
        
        # 跟随系数 = 跟随时刻成交量 / 优势时刻成交量
        follow_coefficient = follow_volume / advantage_volume
        follow_coefficients.append(follow_coefficient)
    
    return follow_coefficients


def process_single_stock_a10(inst: str, df_min: pd.DataFrame,
                              top_n: int = TOP_N_MASSIVE,
                              min_interval: int = MIN_INTERVAL_MINUTES,
                              follow_window: int = FOLLOW_WINDOW) -> Optional[pd.DataFrame]:
    """
    单只股票A10因子处理（严格按研报原文）
    
    计算步骤：
    1. 剔除9:45之前的数据
    2. 识别海量时刻（成交量最大的10个）
    3. 时间间隔过滤，得到优势时刻
    4. 计算每个优势时刻的跟随系数
    5. 日跟随系数 = 日内所有跟随系数的均值
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < 10:
            return None
        
        df_min = df_min.sort_values('date').copy()
        
        daily_results = []
        
        # 按交易日分组处理
        for trade_date, day_df in df_min.groupby('trade_date'):
            if len(day_df) < 10:
                continue
            
            day_df = day_df.sort_values('date').reset_index(drop=True)
            
            # Step 1: 剔除9:45之前的数据
            day_df = filter_by_market_open(day_df)
            
            if len(day_df) < top_n:
                continue
            
            # Step 2: 识别海量时刻（成交量最大的10个）
            day_df = identify_massive_moments_topn(day_df, top_n)
            
            # Step 3: 筛选优势时刻（时间间隔过滤）
            day_df = filter_advantage_moments(day_df, min_interval)
            
            # Step 4: 计算跟随系数
            follow_coefficients = calculate_follow_coefficients(day_df, follow_window)
            
            if len(follow_coefficients) == 0:
                continue
            
            # Step 5: 日跟随系数 = 跟随系数的均值
            daily_follow_coefficient = np.mean(follow_coefficients)
            
            daily_results.append({
                'date': pd.to_datetime(trade_date),
                'instrument': inst,
                'daily_follow_coefficient': daily_follow_coefficient,
                'advantage_count': day_df['is_advantage'].sum(),
                'follow_coeff_mean': daily_follow_coefficient,
                'follow_coeff_std': np.std(follow_coefficients) if len(follow_coefficients) > 1 else 0
            })
        
        if not daily_results:
            return None
        
        return pd.DataFrame(daily_results)
        
    except Exception as e:
        return None


def parallel_process_batch_a10(stock_data_dict: Dict[str, pd.DataFrame],
                                max_workers: int = MAX_WORKERS,
                                top_n: int = TOP_N_MASSIVE,
                                min_interval: int = MIN_INTERVAL_MINUTES,
                                follow_window: int = FOLLOW_WINDOW) -> List[pd.DataFrame]:
    """并行处理一批股票（A10因子）"""
    if not stock_data_dict:
        return []
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_inst = {
            executor.submit(
                process_single_stock_a10, 
                inst, 
                df,
                top_n,
                min_interval,
                follow_window
            ): inst 
            for inst, df in stock_data_dict.items()
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


def calculate_daily_factor_a10(start_date: str, end_date: str,
                                batch_size: int = CHUNK_SIZE,
                                use_parallel: bool = True,
                                top_n: int = TOP_N_MASSIVE,
                                min_interval: int = MIN_INTERVAL_MINUTES,
                                follow_window: int = FOLLOW_WINDOW) -> pd.DataFrame:
    """
    高性能日频因子计算（A10待著而救）
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
            batch_results = parallel_process_batch_a10(
                stock_groups, MAX_WORKERS, top_n, min_interval, follow_window
            )
        else:
            batch_results = []
            for inst, df in stock_groups.items():
                result = process_single_stock_a10(inst, df, top_n, min_interval, follow_window)
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


def calculate_monthly_factor_a10(df_daily: pd.DataFrame, 
                                  window_days: int = FACTOR_WINDOW) -> pd.DataFrame:
    """
    计算A10月因子（月均 + 月稳，等权合成）
    
    研报原文：
    "分别计算过去20天的'日跟随系数'的均值和标准差，
     分别记为'月均待著而救'因子和'月稳待著而救'因子，
     最后再将二者等权合成为'待著而救'因子"
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只，"
          f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 向量化滚动计算
    min_periods = min(10, window_days)
    print(f"  滚动窗口: {window_days}日, 最小有效天数: {min_periods}日")
    
    # 月均待著而救：过去20日跟随系数的均值
    df['monthly_mean'] = df.groupby('instrument')['daily_follow_coefficient'].transform(
        lambda x: x.rolling(window=window_days, min_periods=min_periods).mean()
    )
    
    # 月稳待著而救：过去20日跟随系数的标准差
    df['monthly_std'] = df.groupby('instrument')['daily_follow_coefficient'].transform(
        lambda x: x.rolling(window=window_days, min_periods=min_periods).std()
    )
    
    # 等权合成：待著而救因子 = 月均 + 月稳
    df['final_factor'] = (df['monthly_mean'] + df['monthly_std']) / 2
    
    # 只保留有效因子值
    df_result = df[df['final_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    print(f"  月均因子均值: {df_result['monthly_mean'].mean():.6f}")
    print(f"  月稳因子均值: {df_result['monthly_std'].mean():.6f}")
    print(f"  合成因子均值: {df_result['final_factor'].mean():.6f}")
    
    return df_result[['date', 'instrument', 'final_factor', 'monthly_mean', 'monthly_std']]


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
    
    print(f"=== {FACTOR_NAME}（待著而救）因子计算 - v2.2 BUG修复版 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"\nA10因子参数（严格按研报原文，v2.2修复）:")
    print(f"  - 海量时刻: 成交量最大的{TOP_N_MASSIVE}个时刻")
    print(f"  - 数据预处理: 剔除9:45之前数据")
    print(f"  - 最小间隔: {MIN_INTERVAL_MINUTES}分钟")
    print(f"  - 跟随窗口: {FOLLOW_WINDOW}分钟（仅取优势时刻之后，不含自身）")
    print(f"  - 跟随系数口径: volume（成交量），与研报原文对齐")
    print(f"  - 月因子合成: 月均 + 月稳（等权）")
    
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
    
    # 步骤1：计算日频因子
    print("\n【步骤1】计算日频跟随系数（并行优化）...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor_a10(
        effective_start, end_date,
        batch_size=CHUNK_SIZE,
        use_parallel=use_parallel,
        top_n=TOP_N_MASSIVE,
        min_interval=MIN_INTERVAL_MINUTES,
        follow_window=FOLLOW_WINDOW
    )
    calc_time = time.time() - start_time_calc
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        print(f"  优势时刻统计: 平均{df_daily['advantage_count'].mean():.2f}个/日")
        
        # 步骤2：计算月因子（月均 + 月稳，等权合成）
        print("\n【步骤2】计算月因子（月均+月稳，等权合成）...")
        start_time_factor = time.time()
        df_factor = calculate_monthly_factor_a10(df_daily, window_days=FACTOR_WINDOW)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
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
                print(f"\nA10待著而救因子 v2.2 - 严格按研报原文（跟随系数: volume口径，跟随窗口: 不含优势时刻本身，优势时刻筛选: 与前一个海量时刻比较）")
                print(f"目标Rank IC: -9.28%")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")

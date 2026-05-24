# A06 飞蛾扑火因子 - 高性能优化版 v2.2（8核并行 + 完整复现 + 修复版）
# 来源：方正证券《个股股价跳跃及其对振幅因子的改进》（2022-09-22）
# 档案编号：FQ-20260403-006
# 版本：v2.2 - 修复版（异常处理 + 除零保护）
#
# 【v2.2修复内容】
#   1. 异常处理优化：parallel_process_batch() 中 except: pass → print(f"股票{inst}失败: {e}")
#   2. 除零保护：振幅计算 df['prev_close'].replace(0, np.nan) 避免除以0产生inf
#
# 【v2.1保留内容】
#   1. 完整复现研报逻辑（修正振幅1+2合成）
#   2. 多进程并行优化（8核）
#   3. 向量化滚动计算

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
FACTOR_NAME = "A06"                # 本因子编号（飞蛾扑火）
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
    
    if 'moth_to_flame_factor' in df.columns:
        df.rename(columns={'moth_to_flame_factor': factor_name}, inplace=True)
    
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
        import traceback
        traceback.print_exc()
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

def calculate_daily_jumpiness_vectorized(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    向量化计算日跳跃度（泰勒残项）
    
    构建步骤：
    1) 单利收益率 = 收盘价/前一分钟收盘价 - 1
    2) 连续复利收益率 = ln(收盘价/前一分钟收盘价)
    3) 单复利差 = 单利收益率 - 连续复利收益率
    4) 泰勒残项 = 2 × 单复利差 - 连续复利收益率²
    5) 日跳跃度 = 日内泰勒残项均值
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 向量化计算收益率
    df['simple_return'] = df.groupby(['instrument', 'trade_date'])['close'].pct_change()
    df['prev_close'] = df.groupby(['instrument', 'trade_date'])['close'].shift(1)
    df['log_return'] = np.log(df['close'] / df['prev_close'])
    
    # 计算单复利差和泰勒残项（向量化）
    df['diff_return'] = df['simple_return'] - df['log_return']
    df['taylor_residual'] = 2 * df['diff_return'] - df['log_return'] ** 2
    
    # 按日和股票聚合
    daily_agg = df.groupby(['instrument', 'trade_date']).agg({
        'taylor_residual': 'mean',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    daily_agg.columns = ['instrument', 'trade_date', 'daily_jumpiness', 'daily_high', 'daily_low', 'daily_close']
    daily_agg['date'] = pd.to_datetime(daily_agg['trade_date'])
    
    return daily_agg


def process_single_stock_moth_to_flame(inst: str, df_min: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    单只股票飞蛾扑火因子计算（优化版）- 处理多日期数据
    【v2.1】保留日频最高/最低价用于修正振幅2计算
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < 10:
            return None
        
        df_min = df_min.sort_values('date').copy()
        df_min['trade_date'] = pd.to_datetime(df_min['date']).dt.date
        
        # 向量化计算日跳跃度
        df_min['simple_return'] = df_min.groupby('trade_date')['close'].pct_change()
        df_min['prev_close'] = df_min.groupby('trade_date')['close'].shift(1)
        df_min['log_return'] = np.log(df_min['close'] / df_min['prev_close'])
        df_min['diff_return'] = df_min['simple_return'] - df_min['log_return']
        df_min['taylor_residual'] = 2 * df_min['diff_return'] - df_min['log_return'] ** 2
        
        # 按日聚合
        daily_results = []
        for trade_date, group in df_min.groupby('trade_date'):
            group_valid = group[group['taylor_residual'].notna()]
            if len(group_valid) < 5:
                continue
            
            daily_jumpiness = group_valid['taylor_residual'].mean()
            daily_high = group_valid['high'].max()
            daily_low = group_valid['low'].min()
            daily_close = group_valid['close'].iloc[-1]
            
            daily_results.append({
                'date': pd.to_datetime(trade_date),
                'instrument': inst,
                'daily_jumpiness': daily_jumpiness,
                'daily_high': daily_high,
                'daily_low': daily_low,
                'daily_close': daily_close
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
            executor.submit(process_single_stock_moth_to_flame, inst, df): inst 
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
                result = process_single_stock_moth_to_flame(inst, df)
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


def calculate_corrected_amplitude_2_vectorized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    【v2.1新增】修正振幅因子2（基于日频最高/最低价）- 向量化实现
    
    研报原文（第9页）：
    1) 使用"单利"和"连续复利"计算从t-1日最低价到t日最高价的收益率
    2) 泰勒残项 = 2×单复利差 - 连续复利收益率²
    3) 截面均值分类：泰勒残项 < 均值→太阳型(振幅×-1)，泰勒残项 > 均值→火把型(振幅×1)
    4) 修正振幅2 = 20日翻转振幅均值
    """
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 获取前一日最低价
    df['prev_low'] = df.groupby('instrument')['daily_low'].shift(1)
    
    # 步骤1：计算从t-1日最低价到t日最高价的收益率
    # 单利收益率 = t日最高价/t-1日最低价 - 1
    df['simple_return_hl'] = df['daily_high'] / df['prev_low'] - 1
    
    # 连续复利收益率 = ln(t日最高价/t-1日最低价)
    df['log_return_hl'] = np.log(df['daily_high'] / df['prev_low'])
    
    # 步骤2：计算泰勒残项 = 2×(单利-复利) - 复利²
    df['diff_return_hl'] = df['simple_return_hl'] - df['log_return_hl']
    df['taylor_residual_hl'] = 2 * df['diff_return_hl'] - df['log_return_hl'] ** 2
    
    # 步骤3：计算每日截面均值（所有股票的泰勒残项均值）- 向量化
    df['cross_mean_residual_hl'] = df.groupby('date')['taylor_residual_hl'].transform('mean')
    
    # 步骤4：根据泰勒残项截面均值分类，翻转振幅
    # 泰勒残项 < 截面均值：太阳型（振幅 × -1）
    # 泰勒残项 > 截面均值：火把型（振幅 × 1）
    df['flipped_amplitude_2'] = np.where(
        df['taylor_residual_hl'] < df['cross_mean_residual_hl'],
        -df['amplitude'],  # 太阳型：翻转
        df['amplitude']    # 火把型：不变
    )
    
    return df


def calculate_moth_to_flame_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算飞蛾扑火因子（向量化优化版）- 【v2.1完整复现】
    
    【v2.1修正】合成公式：
    1) 月跳跃度 = (月均跳跃度 + 月稳跳跃度) / 2
    2) 修正振幅1 = 基于分钟数据日跳跃度的20日翻转振幅均值
    3) 修正振幅2 = 基于日频最高/最低价的20日翻转振幅均值 【新增】
    4) 修正振幅 = (修正振幅1 + 修正振幅2) / 2 【修正，等权合成】
    5) 飞蛾扑火因子 = (月跳跃度 + 修正振幅) / 2
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只，日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 计算前一日收盘价和振幅
    df['prev_close'] = df.groupby('instrument')['daily_close'].shift(1)
    # 【v2.2修复】增加除零保护，prev_close为0时设为NaN避免inf
    df['amplitude'] = (df['daily_high'] - df['daily_low']) / df['prev_close'].replace(0, np.nan)
    
    # 【v2.1新增】计算修正振幅因子2
    df = calculate_corrected_amplitude_2_vectorized(df)
    
    # 计算日跳跃度截面均值用于修正振幅1分类（向量化）
    df['jumpiness_mean_cross'] = df.groupby('date')['daily_jumpiness'].transform('mean')
    
    # 修正振幅1：根据日跳跃度截面均值分类翻转振幅（向量化）
    df['flipped_amplitude_1'] = np.where(
        df['daily_jumpiness'] < df['jumpiness_mean_cross'],
        -df['amplitude'],  # 太阳型：翻转
        df['amplitude']     # 火把型：不变
    )
    
    # 向量化滚动计算
    min_periods = min(10, FACTOR_WINDOW)
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    # 月跳跃度计算
    df['monthly_jumpiness_mean'] = df.groupby('instrument')['daily_jumpiness'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    df['monthly_jumpiness_std'] = df.groupby('instrument')['daily_jumpiness'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
    )
    
    # 月跳跃度 = (月均 + 月稳) / 2
    df['monthly_jumpiness'] = (df['monthly_jumpiness_mean'] + df['monthly_jumpiness_std']) / 2
    
    # 修正振幅1 = 20日翻转振幅均值
    df['corrected_amplitude_1'] = df.groupby('instrument')['flipped_amplitude_1'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 【v2.1新增】修正振幅2 = 20日翻转振幅2均值
    df['corrected_amplitude_2'] = df.groupby('instrument')['flipped_amplitude_2'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 【v2.1修正】修正振幅 = (修正振幅1 + 修正振幅2) / 2
    df['corrected_amplitude'] = (df['corrected_amplitude_1'] + df['corrected_amplitude_2']) / 2
    
    # 飞蛾扑火因子 = (月跳跃度 + 修正振幅) / 2
    df['moth_to_flame_factor'] = (df['monthly_jumpiness'] + df['corrected_amplitude']) / 2
    
    # 只保留有效因子值
    df_result = df[df['moth_to_flame_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    
    return df_result[['date', 'instrument', 'moth_to_flame_factor', 'monthly_jumpiness', 
                      'corrected_amplitude', 'corrected_amplitude_1', 'corrected_amplitude_2']]


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
    
    print(f"=== {FACTOR_NAME}（飞蛾扑火）因子计算 - 高性能优化版v2.1（完整复现）===")
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
    
    print("\n【因子逻辑-v2.1完整复现】")
    print("1) 基于分钟数据计算泰勒残项，得到日跳跃度")
    print("2) 修正振幅1：根据日跳跃度截面均值将振幅分类为'太阳型'和'火把型'，翻转后20日均值")
    print("3) 修正振幅2：基于日频最高/最低价计算泰勒残项，截面分类翻转后20日均值 【新增】")
    print("4) 修正振幅 = (修正振幅1 + 修正振幅2) / 2 【修正】")
    print("5) 飞蛾扑火因子 = (月跳跃度 + 修正振幅) / 2")
    
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
        
        # 步骤2：计算飞蛾扑火因子
        print("\n【步骤2】计算飞蛾扑火因子（向量化滚动）...")
        start_time_factor = datetime.now()
        df_factor = calculate_moth_to_flame_factor_optimized(df_daily)
        factor_time = (datetime.now() - start_time_factor).total_seconds()
        
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
                print(f"\n预期表现: Rank IC ~ -8.90%（研报基准）")
                print(f"多空年化收益 ~ 37.30%")
                print(f"月度胜率 ~ 87.83%")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")

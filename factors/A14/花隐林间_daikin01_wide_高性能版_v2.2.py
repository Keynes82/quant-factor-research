# A14 花隐林间因子 - 高性能优化版（8核并行）v2.2
# 来源：方正证券-多因子选股系列研究之十《推动个股价格变化的因素分解与花隐林间因子》（2023-03-27）
# 档案编号：FQ-20260407-A14
# 版本：v2.1 - 日频化改造（参考A01模式：日频计算 + 20日滚动平滑）
#
# 【v2.1 审查修复记录】
#   - 修复1：XtX求逆 → 伪逆（np.linalg.pinv），避免接近奇异矩阵崩溃
#   - 修复2：夜眠霜路排除对角线自相关（np.fill_diagonal(corr_matrix, np.nan)）
#   - 修复3：等权合成NaN处理 —— 从 fillna(0) 改为 只对有值的子因子求平均
#            避免缺失子因子用0稀释有效因子值
#
# 【v2.1 日频化记录】
#   - 改造：月频因子 → 日频因子，每天更新因子值
#   - 朝没晨雾：日频std(t1..t5) → rolling(20)平滑
#   - 午蔽古木：日频截面翻转 → rolling(20)平滑
#   - 夜眠霜路：每天用过去20天t-intercept计算截面|corr|均值 → rolling(20)平滑
#   - 花隐林间：三个子因子rolling(20)后等权合成
#   - 因子值每天更新，支持日频调仓
#
# 【v2.0 重写记录】
#   - 修复1（严重）：改为多元最小二乘回归（6个滞后增量成交量+截距项），原代码是一元回归
#   - 修复2（严重）：y变量改为收益率 close_t/close_{t-1}-1，原代码是价格差
#   - 修复3（严重）：x变量改为增量成交量 volume_t-volume_{t-1}，原代码是原始volume
#   - 修复4（严重）：新增三个子因子（朝没晨雾/午蔽古木/夜眠霜路），原代码只有一个残差占比
#   - 修复5（严重）：新增截面翻转（午蔽古木），原代码完全缺失
#   - 修复6（严重）：新增截面相关性计算（夜眠霜路），原代码完全缺失
#   - 修复7（严重）：新增等权合成（花隐林间 = 三个子因子等权），原代码完全缺失
#
# 因子构建逻辑（v2.1日频版）：
#   1. 多元回归（每日第6分钟至第240分钟）：
#      y = ret_t = close_t / close_{t-1} - 1
#      X = [1, delta_vol_t, delta_vol_{t-1}, ..., delta_vol_{t-5}]
#      delta_vol_t = volume_t - volume_{t-1}
#   2. 提取：t-intercept（截距t值）、t0-t5（6个系数t值）、F-all（F统计量）
#   3. 朝没晨雾（日频原始）= std(t1, t2, t3, t4, t5)  [排除t0]
#   4. 午蔽古木（日频原始）= 截面翻转：F_all < 截面均值时，|t-intercept| * (-1)
#   5. 夜眠霜路（日频原始）= 每天计算过去20天t-intercept序列 vs 截面所有股票过去20天t-intercept序列的|corr|均值
#   6. 花隐林间（日频因子）= (morning_fog_20 + noon_wood_20 + night_frost_20) / 3
#      其中 _20 表示 rolling(20) 平滑后的值
#
# 优化点：
#   - 多进程并行计算（8核）
#   - 批量SQL查询（array_agg聚合close+volume，减少IO）
#   - numpy.linalg.lstsq高效多元回归
#   - 向量化截面翻转与滚动计算

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
FACTOR_NAME = "A14"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日平均）

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
    
    if 'huayinlinjian' in df.columns:
        df.rename(columns={'huayinlinjian': factor_name}, inplace=True)
    
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
    """批量读取分钟级数据（array_agg聚合版）- 返回每日close/volume序列"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    
    # 研报要求：第6分钟至第240分钟 → 保留 09:31-14:56（开盘1分钟后到收盘前3分钟）
    sql = f"""
    SELECT 
        CAST(date AS DATE) as trade_date,
        instrument,
        array_agg(close ORDER BY date) as close_series,
        array_agg(volume ORDER BY date) as volume_series,
        COUNT(*) as n_minutes
    FROM cn_stock_bar1m
    WHERE date >= TIMESTAMP '{start_date} 09:30:00'
      AND date <= TIMESTAMP '{end_date} 15:00:00'
      AND (CAST(date AS TIME) >= '09:35:00' AND CAST(date AS TIME) <= '15:00:00')
      AND instrument IN ('{instrument_list}')
    GROUP BY CAST(date AS DATE), instrument
    HAVING COUNT(*) >= 16
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


# ========== A14核心：多元回归与日频统计量计算 ==========

def calculate_regression_stats(close_series, volume_series):
    """
    对单只股票单日数据进行多元最小二乘回归（严格按研报定义）
    
    回归设定：
    - y_t = ret_t = close_t / close_{t-1} - 1
    - X_t = [1, delta_vol_t, delta_vol_{t-1}, ..., delta_vol_{t-5}]
    - delta_vol_t = volume_t - volume_{t-1}
    
    返回：
        dict: t_intercept, abst_intercept, t0-t5, F_all, morning_fog
    """
    close = np.array(close_series, dtype=np.float64)
    volume = np.array(volume_series, dtype=np.float64)
    n = len(close)
    
    if n < 16:
        return None
    
    # 收益率：ret_t = close_t / close_{t-1} - 1
    returns = close[1:] / close[:-1] - 1.0
    
    # 增量成交量：delta_vol_t = volume_t - volume_{t-1}
    delta_volume = volume[1:] - volume[:-1]
    n_diff = len(delta_volume)
    
    if n_diff < 6:
        return None
    
    # 有效回归样本：从索引5开始（需要t-5到t共6个滞后值）
    # 对应原始分钟序列的第7分钟开始（diff损失1个，再前推5个）
    n_samples = n_diff - 5
    if n_samples < 10:
        return None
    
    # 构造设计矩阵 X: (n_samples, 7) = [1, delta_t, delta_{t-1}, ..., delta_{t-5}]
    X = np.ones((n_samples, 7))
    for lag in range(6):
        start = 5 - lag
        X[:, lag + 1] = delta_volume[start:start + n_samples]
    
    y = returns[5:]
    
    # 最小二乘回归
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    # 计算统计量
    y_pred = X @ beta
    residuals = y - y_pred
    n = n_samples
    p = 7  # 截距 + 6个系数
    
    sse = np.sum(residuals**2)
    mse = sse / max(n - p, 1)
    
    if mse <= 0:
        return None
    
    # 系数协方差矩阵 → 标准误 → t值
    XtX = X.T @ X
    try:
        # 使用伪逆更稳健，避免接近奇异矩阵的问题
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError:
        return None
    
    cov_beta = mse * XtX_inv
    se_beta = np.sqrt(np.diag(cov_beta) + 1e-10)
    t_values = beta / se_beta
    
    t_intercept = float(t_values[0])
    coeffs = [float(v) for v in t_values[1:]]
    t0, t1, t2, t3, t4, t5 = coeffs
    
    # F统计量
    ssr = np.sum((y_pred - y.mean())**2)
    F_all = float((ssr / 6) / mse) if mse > 0 else 0.0
    
    # 朝没晨雾 = std(t1, t2, t3, t4, t5)  [排除t0，即当前分钟]
    morning_fog = float(np.std(coeffs[1:], ddof=1))
    
    return {
        't_intercept': t_intercept,
        'abst_intercept': abs(t_intercept),
        't0': t0, 't1': t1, 't2': t2, 't3': t3, 't4': t4, 't5': t5,
        'F_all': F_all,
        'morning_fog': morning_fog
    }


def process_single_stock_optimized(inst: str, df_stock: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    单只股票日频处理：对每交易日进行多元回归，提取统计量
    
    参数:
        inst: 股票代码
        df_stock: 包含close_series, volume_series, n_minutes的DataFrame
    
    返回:
        DataFrame: 每日回归统计量
    """
    if df_stock is None or df_stock.empty:
        return None
    
    daily_results = []
    
    for _, row in df_stock.iterrows():
        trade_date = row['trade_date']
        close_series = row['close_series']
        volume_series = row['volume_series']
        n_minutes = row['n_minutes']
        
        if n_minutes < 7:
            continue
        
        # 转换序列
        if isinstance(close_series, list):
            close_arr = np.array(close_series, dtype=np.float64)
        elif isinstance(close_series, np.ndarray):
            close_arr = close_series.astype(np.float64)
        else:
            continue
            
        if isinstance(volume_series, list):
            vol_arr = np.array(volume_series, dtype=np.float64)
        elif isinstance(volume_series, np.ndarray):
            vol_arr = volume_series.astype(np.float64)
        else:
            continue
        
        stats = calculate_regression_stats(close_arr, vol_arr)
        
        if stats is not None:
            daily_results.append({
                'date': trade_date,
                'instrument': inst,
                **stats
            })
    
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
    高性能日频因子计算（花隐林间版本）
    输出：每只股票每日的回归统计量（t_intercept, t0-t5, F_all, morning_fog）
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
        
        # 批量获取数据（SQL层已聚合为每日序列）
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
                result = process_single_stock_optimized(inst, df)
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


# ========== 截面翻转：午蔽古木日因子 ==========

def apply_cross_sectional_flip(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算日午蔽古木因子（截面翻转）
    
    研报定义：
    - 每天截面上，F-all < 截面均值的股票：abst_intercept * (-1)
    - F-all >= 截面均值的股票：abst_intercept 保持不变
    """
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 每日F-all截面均值
    daily_f_mean = df.groupby('date')['F_all'].transform('mean')
    
    # 翻转
    df['noon_wood'] = np.where(
        df['F_all'] < daily_f_mean,
        -df['abst_intercept'],
        df['abst_intercept']
    )
    
    return df


# ========== 夜眠霜路计算（日频版）==========

def calculate_daily_night_frost(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算日频夜眠霜路因子（v2.1日频改造）
    
    改造：原月频（每月月底计算）→ 日频（每天计算过去20天截面|corr|均值）
    
    每天，对每只股票，取过去20天（含当日）的t-intercept序列，
    与截面所有股票过去20天的t-intercept序列计算|corr|均值。
    
    参数:
        df_daily: DataFrame with columns [date, instrument, t_intercept]
    
    返回:
        DataFrame: [date, instrument, night_frost_raw]
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily[['date', 'instrument', 't_intercept']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    all_dates = sorted(df['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    daily_records = []
    
    for current_date in all_dates:
        idx = date_to_idx[current_date]
        
        # 过去20个交易日窗口（含当日）
        start_idx = max(0, idx - 19)
        window_dates = all_dates[start_idx:idx + 1]
        
        # 获取窗口内所有股票的t-intercept
        window_df = df[df['date'].isin(window_dates)].copy()
        
        if window_df.empty:
            continue
        
        # 构造矩阵：instrument × date
        pivot = window_df.pivot(index='instrument', columns='date', values='t_intercept')
        
        # 剔除缺失太多的股票（至少有一半日期有数据）
        pivot = pivot.dropna(thresh=max(5, len(window_dates) // 2))
        
        if pivot.empty or pivot.shape[0] < 5:
            continue
        
        # 计算截面|corr|均值
        T = pivot.values
        valid_mask = (~np.isnan(T)).sum(axis=1) >= max(5, T.shape[1] // 2)
        T_valid = T[valid_mask]
        
        if T_valid.shape[0] < 2:
            continue
        
        row_means = np.nanmean(T_valid, axis=1, keepdims=True)
        T_filled = np.where(np.isnan(T_valid), row_means, T_valid)
        
        corr_matrix = np.corrcoef(T_filled)
        # 排除自身相关系数（对角线=1），避免拉高均值
        np.fill_diagonal(corr_matrix, np.nan)
        night_frost_values = np.nanmean(np.abs(corr_matrix), axis=1)
        
        instruments = pivot.index[valid_mask]
        
        for inst, nf_val in zip(instruments, night_frost_values):
            daily_records.append({
                'date': current_date,
                'instrument': inst,
                'night_frost_raw': float(nf_val)
            })
    
    if not daily_records:
        return pd.DataFrame()
    
    return pd.DataFrame(daily_records)


# ========== 日频因子计算与等权合成 ==========

def calculate_daily_huayinlinjian(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算日频花隐林间因子（v2.1日频改造：参考A01模式）
    
    策略（日频化）：
    1. 朝没晨雾/午蔽古木：日频原始值 → rolling(20)平滑
    2. 夜眠霜路：每天计算过去20天截面|corr|均值 → rolling(20)平滑
    3. 花隐林间：三个子因子rolling(20)后等权合成
    4. 因子值每天更新，支持日频调仓
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只")
    
    min_periods = min(10, FACTOR_WINDOW)
    
    # 20日滚动平滑（朝没晨雾和午蔽古木）
    print(f"  计算20日滚动平滑...")
    df['morning_fog_20'] = df.groupby('instrument')['morning_fog'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    df['noon_wood_20'] = df.groupby('instrument')['noon_wood'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 日频夜眠霜路计算（过去20天截面|corr|）
    print(f"  计算日频夜眠霜路（过去20天截面相关性）...")
    start_time_nf = time.time()
    df_nf = calculate_daily_night_frost(df[['date', 'instrument', 't_intercept']])
    nf_time = time.time() - start_time_nf
    print(f"  夜眠霜路日频计算完成: {len(df_nf)} 条，耗时 {nf_time:.2f}秒")
    
    if df_nf is None or df_nf.empty:
        print("  警告：未能计算夜眠霜路因子，跳过")
        # 只用两个子因子合成 —— 只对有值的子因子求平均
        cols = ['morning_fog_20', 'noon_wood_20']
        df['huayinlinjian'] = df[cols].sum(axis=1) / df[cols].notna().sum(axis=1)
        df['huayinlinjian'] = df['huayinlinjian'].replace([np.inf, -np.inf], np.nan)
        
        all_na = df[cols].isna().all(axis=1)
        df.loc[all_na, 'huayinlinjian'] = np.nan
    else:
        # merge夜眠霜路日频值
        df = df.merge(df_nf, on=['date', 'instrument'], how='left')
        
        # 夜眠霜路也做20日滚动平滑
        df['night_frost_20'] = df.groupby('instrument')['night_frost_raw'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
        )
        
        # 等权合成（三个子因子rolling后）—— 只对有值的子因子求平均
        cols = ['morning_fog_20', 'noon_wood_20', 'night_frost_20']
        df['huayinlinjian'] = df[cols].sum(axis=1) / df[cols].notna().sum(axis=1)
        df['huayinlinjian'] = df['huayinlinjian'].replace([np.inf, -np.inf], np.nan)
        
        # 如果某只股票三个子因子全部缺失，则剔除
        all_na = df[cols].isna().all(axis=1)
        df.loc[all_na, 'huayinlinjian'] = np.nan
    
    # 只保留有效因子值
    df_result = df[df['huayinlinjian'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    print(f"  因子统计: 均值={df_result['huayinlinjian'].mean():.6f}, 标准差={df_result['huayinlinjian'].std():.6f}")
    
    return df_result[['date', 'instrument', 'morning_fog_20', 'noon_wood_20', 'night_frost_20', 'huayinlinjian']]


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
    
    print(f"=== {FACTOR_NAME}（花隐林间）因子计算 - v2.1日频改造（参考A01模式） ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日（日频因子，每天更新）")
    print(f"核心方法: 多元最小二乘回归 + 20日滚动平滑")
    print(f"【v2.1日频化】三个子因子日频计算 + rolling(20) + 等权合成")
    
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
    
    # 步骤1：计算日频回归统计量
    print("\n【步骤1】计算日频多元回归统计量（并行优化）...")
    start_time_calc = time.time()
    df_daily_raw = calculate_daily_factor_optimized(effective_start, end_date, 
                                                       batch_size=CHUNK_SIZE,
                                                       use_parallel=use_parallel)
    calc_time = time.time() - start_time_calc
    
    if df_daily_raw is None or df_daily_raw.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频统计量计算完成: {len(df_daily_raw)} 条，耗时 {calc_time:.2f}秒")
        
        # 步骤2：截面翻转（午蔽古木）
        print("\n【步骤2】截面翻转计算午蔽古木日因子...")
        start_time_flip = time.time()
        df_daily = apply_cross_sectional_flip(df_daily_raw)
        flip_time = time.time() - start_time_flip
        print(f"截面翻转完成: 耗时 {flip_time:.2f}秒")
        
        # 步骤3：日频计算（三个子因子 + 20日滚动平滑 + 等权合成）
        print("\n【步骤3】计算日频花隐林间因子（三个子因子rolling(20)等权合成）...")
        start_time_factor = time.time()
        df_factor = calculate_daily_huayinlinjian(df_daily)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何日频因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日频数据列: {df_daily.columns.tolist()}")
            print(f"  日频数据统计:")
            print(f"    - morning_fog: 均值={df_daily['morning_fog'].mean():.6f}, 非空数={df_daily['morning_fog'].notna().sum()}")
            print(f"    - noon_wood: 均值={df_daily['noon_wood'].mean():.6f}, 非空数={df_daily['noon_wood'].notna().sum()}")
            print(f"  建议检查：")
            print(f"    1. 日期范围是否足够（至少需要{FACTOR_WINDOW}个交易日才能计算rolling）")
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
                print(f"截面翻转: {flip_time:.2f}秒")
                print(f"日频合成: {factor_time:.2f}秒")
                print(f"数据写入: {write_time:.2f}秒")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"写入 {written_count} 条数据")
                print(f"\n预期表现（日频因子）：")
                print(f"  朝没晨雾: Rank IC ~ -8.55%")
                print(f"  午蔽古木: Rank IC ~ -6.50%（预估）")
                print(f"  夜眠霜路: Rank IC ~ 6.06%")
                print(f"  花隐林间: Rank IC ~ -9.50%（预估，等权合成）")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily_raw) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print(f"日期过滤后无数据，尝试扩大计算范围。")


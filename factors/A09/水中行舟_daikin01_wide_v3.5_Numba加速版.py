# A09 水中行舟因子 - v3.5 Numba加速版（daikin01）
# 来源：方正证券《个股成交额的市场跟随性与"水中行舟"因子》（2023-02-15）
# 档案编号：FQ-20260403-009
# 因子名称：A09（水中行舟）
# 版本：v3.5 - Numba加速高低额差计算，日期级并行
#
# 【v3.5优化内容】
#   1. 【优化四】高低额差：逐日Pandas循环 → Numba日期级并行（prange）
#      绕过DataFrame创建/merge开销，16核同时处理不同日期
#      预期加速：5-15倍
#   2. 【优化一】随波逐流因子：逐对循环spearmanr → 矩阵运算（批量rank + 矩阵乘法）
#      复杂度从 O(m² × n log n) 降到 O(m × n log n + m² × n)，加速10-50倍
#   3. 【优化二】孤雁出群因子：逐对循环pearsonr → np.corrcoef矩阵运算，加速5-10倍
#   4. 【优化三】移除ProcessPoolExecutor多进程，避免大数据序列化开销
#      改用向量化单线程，进程启动/通信开销降为0
#   5. 保留v3.3所有修正逻辑（日期对齐、无双重平滑、截面均值填充）
#
# 因子构建逻辑（严格按研报定义）：
#   【随波逐流因子】- 正向因子
#   【孤雁出群因子】- 负向因子
#   【合成】水中行舟因子 = (随波逐流 + 孤雁出群) / 2

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Dict, Optional
from scipy import stats
import numba
import warnings
warnings.filterwarnings('ignore')

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"
FACTOR_NAME = "A09"
SAFETY_BUFFER_DAYS = 30
FACTOR_WINDOW = 20
REASONABLE_RETURN_WINDOW = 20
MIN_STOCKS_FOR_CORR = 50


def get_last_computed_date(table_id: str, factor_name: str = FACTOR_NAME) -> Optional[str]:
    """获取该因子最后计算的日期，用于增量计算。"""
    try:
        ds = dai.DataSource(table_id)
        df = ds.read_bdb(as_type=pd.DataFrame)
        if not df.empty and factor_name in df.columns:
            factor_df = df[df[factor_name].notna()]
            if not factor_df.empty:
                last_date = pd.to_datetime(factor_df['date'].max())
                extended_date = last_date - timedelta(days=SAFETY_BUFFER_DAYS)
                result = extended_date.strftime('%Y-%m-%d')
                print(f"检测到历史数据，最后计算日期: {last_date.date()}，增量起始: {result}")
                return result
    except Exception as e:
        print(f"读取历史数据失败（可能是首次计算）: {e}")
    return None


def prepare_factor_df_for_write(df: pd.DataFrame, factor_name: str = FACTOR_NAME) -> pd.DataFrame:
    """规范因子DataFrame，转换为宽表格式。"""
    if df is None or df.empty:
        return pd.DataFrame()
    if 'shui_zhong_xing_zhou' in df.columns:
        df = df.rename(columns={'shui_zhong_xing_zhou': factor_name})
    required_cols = ['date', 'instrument', factor_name]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['instrument'] = df['instrument'].astype(str).str.strip()
    df[factor_name] = pd.to_numeric(df[factor_name], errors='coerce')
    df_out = df[required_cols].copy()
    df_out = df_out.dropna().reset_index(drop=True)
    df_out = df_out.sort_values(['date', 'instrument']).drop_duplicates(
        subset=['date', 'instrument'], keep='last'
    ).reset_index(drop=True)
    return df_out


def safe_write_to_library(df: pd.DataFrame, table_id: str = FACTOR_LIBRARY_TABLE,
                          factor_name: str = FACTOR_NAME, overwrite: bool = False) -> int:
    """将因子数据写入宽表。"""
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
            data=df_combined, id=table_id,
            unique_together=["date", "instrument"], on_duplicates="last"
        )
        print(f"成功写入 '{factor_name}'：{len(df)} 行到 '{table_id}'")
        return len(df)
    except Exception as e:
        print(f"写入失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def fetch_daily_data_for_reasonable_return(start_date: str, end_date: str) -> pd.DataFrame:
    """获取日频数据用于计算合理收益率。"""
    start_dt = (pd.to_datetime(start_date) - pd.DateOffset(days=REASONABLE_RETURN_WINDOW + 5)).strftime('%Y-%m-%d')
    sql = f"""
    SELECT CAST(date AS DATE) AS trade_date, instrument, open, close
    FROM cn_stock_bar1d
    WHERE date >= '{start_dt}' AND date <= '{end_date}'
    ORDER BY instrument, trade_date
    """
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        df['instrument'] = df['instrument'].astype(str)
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['intraday_return'] = df['close'] / df['open'] - 1
        return df
    except Exception as e:
        print(f"日频数据查询失败: {e}")
        return pd.DataFrame()


def calculate_reasonable_return(df_daily: pd.DataFrame) -> pd.DataFrame:
    """计算合理收益率 = 过去20日日内收益率的均值。"""
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'trade_date']).reset_index(drop=True)
    df['reasonable_return'] = df.groupby('instrument')['intraday_return'].transform(
        lambda x: x.rolling(window=REASONABLE_RETURN_WINDOW, min_periods=10).mean()
    )
    return df


def fetch_minute_data_with_market(start_date: str, end_date: str) -> pd.DataFrame:
    """获取分钟级数据。"""
    sql = f"""
    SELECT date, open, high, low, close, volume, amount, instrument
    FROM cn_stock_bar1m
    WHERE date >= TIMESTAMP '{start_date} 09:30:00'
      AND date <= TIMESTAMP '{end_date} 15:00:00'
    """
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df['trade_date'] = df['date'].dt.date
        df['instrument'] = df['instrument'].astype(str)
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"分钟数据查询失败: {e}")
        return pd.DataFrame()


def fetch_daily_market_cap(start_date: str, end_date: str) -> pd.DataFrame:
    """获取流通市值数据。"""
    sql = f"""
    SELECT CAST(date AS DATE) AS trade_date, instrument, float_market_cap AS circulating_market_cap
    FROM cn_stock_prefactors
    WHERE date >= '{start_date}' AND date <= '{end_date}'
    """
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        df['instrument'] = df['instrument'].astype(str)
        df['circulating_market_cap'] = pd.to_numeric(df['circulating_market_cap'], errors='coerce')
        return df
    except Exception as e:
        print(f"流通市值查询失败: {e}")
        return pd.DataFrame()


# ========== 优化四：Numba加速高低额差 ==========

@numba.njit(parallel=True)
def _calc_high_low_numba(
    day_ids, stock_ids, opens, closes, amounts,
    day_start, day_end,
    rr_matrix, cap_matrix,
    n_days, n_stocks
):
    """
    日期级并行计算高低额差。
    prange让多核各处理不同的日期。
    
    输入：
      day_ids: 每条分钟数据的日期ID (int32)
      stock_ids: 每条分钟数据的股票ID (int32)
      opens, closes, amounts: 分钟数据 (float64)
      day_start[d], day_end[d]: 日期d在数组中的起止索引
      rr_matrix[d, s]: 日期d股票s的合理收益率
      cap_matrix[d, s]: 日期d股票s的流通市值
    输出：
      high_low_diff[d, s]: 日期d股票s的高低额差
    """
    high_low_diff = np.full((n_days, n_stocks), np.nan, dtype=np.float64)
    
    for d in numba.prange(n_days):
        start = day_start[d]
        end = day_end[d]
        if end <= start:
            continue
        
        # Step 1: 找每只股票当天的第一行开盘价
        first_open = np.full(n_stocks, np.nan, dtype=np.float64)
        for i in range(start, end):
            s = stock_ids[i]
            if np.isnan(first_open[s]):
                first_open[s] = opens[i]
        
        # Step 2: 累加高位/低位成交额
        high_amount = np.zeros(n_stocks, dtype=np.float64)
        low_amount = np.zeros(n_stocks, dtype=np.float64)
        
        for i in range(start, end):
            s = stock_ids[i]
            fo = first_open[s]
            if np.isnan(fo):
                continue
            
            rel_return = closes[i] / fo - 1.0
            rr = rr_matrix[d, s]
            if np.isnan(rr):
                continue
            
            if rel_return > rr:
                high_amount[s] += amounts[i]
            elif rel_return < rr:
                low_amount[s] += amounts[i]
        
        # Step 3: 计算高低额差
        for s in range(n_stocks):
            cap = cap_matrix[d, s]
            if not np.isnan(first_open[s]) and cap > 0:
                high_low_diff[d, s] = (high_amount[s] - low_amount[s]) / cap
    
    return high_low_diff


def _encode_and_prepare_high_low(df_minute, df_daily, market_cap_df):
    """
    预处理：把Pandas数据转成Numba友好的NumPy数组格式。
    """
    # 编码日期和股票为连续整数
    unique_dates = sorted(df_minute['trade_date'].unique())
    unique_stocks = sorted(df_minute['instrument'].unique())
    date_to_id = {d: i for i, d in enumerate(unique_dates)}
    stock_to_id = {s: i for i, s in enumerate(unique_stocks)}
    n_days = len(unique_dates)
    n_stocks = len(unique_stocks)
    
    # 分钟数据编码
    df_minute = df_minute.sort_values(['trade_date', 'instrument', 'date']).copy()
    df_minute['day_id'] = df_minute['trade_date'].map(date_to_id).astype(np.int32)
    df_minute['stock_id'] = df_minute['instrument'].map(stock_to_id).astype(np.int32)
    
    # 计算每天的start/end索引
    day_counts = df_minute.groupby('day_id').size().values.astype(np.int32)
    day_start = np.zeros(n_days, dtype=np.int32)
    day_end = np.zeros(n_days, dtype=np.int32)
    pos = 0
    for d in range(n_days):
        day_start[d] = pos
        day_end[d] = pos + day_counts[d]
        pos += day_counts[d]
    
    # 合理收益率矩阵 (day × stock)
    rr_matrix = np.full((n_days, n_stocks), np.nan, dtype=np.float64)
    for _, row in df_daily.iterrows():
        d = date_to_id.get(row['trade_date'])
        s = stock_to_id.get(row['instrument'])
        if d is not None and s is not None:
            rr_matrix[d, s] = row['reasonable_return']
    
    # 流通市值矩阵 (day × stock)
    cap_matrix = np.full((n_days, n_stocks), np.nan, dtype=np.float64)
    for _, row in market_cap_df.iterrows():
        d = date_to_id.get(row['trade_date'])
        s = stock_to_id.get(row['instrument'])
        if d is not None and s is not None:
            cap_matrix[d, s] = row['circulating_market_cap']
    
    # 提取NumPy数组
    day_ids = df_minute['day_id'].values.astype(np.int32)
    stock_ids = df_minute['stock_id'].values.astype(np.int32)
    opens = df_minute['open'].values.astype(np.float64)
    closes = df_minute['close'].values.astype(np.float64)
    amounts = df_minute['amount'].values.astype(np.float64)
    
    return {
        'day_ids': day_ids, 'stock_ids': stock_ids,
        'opens': opens, 'closes': closes, 'amounts': amounts,
        'day_start': day_start, 'day_end': day_end,
        'rr_matrix': rr_matrix, 'cap_matrix': cap_matrix,
        'n_days': n_days, 'n_stocks': n_stocks,
        'unique_dates': unique_dates, 'unique_stocks': unique_stocks,
    }


def calc_high_low_numba_wrapper(prep):
    """
    封装：调用Numba函数，结果转DataFrame。
    """
    result_matrix = _calc_high_low_numba(
        prep['day_ids'], prep['stock_ids'],
        prep['opens'], prep['closes'], prep['amounts'],
        prep['day_start'], prep['day_end'],
        prep['rr_matrix'], prep['cap_matrix'],
        prep['n_days'], prep['n_stocks']
    )
    
    # 结果转DataFrame（只保留非NaN）
    results = []
    for d in range(prep['n_days']):
        date = prep['unique_dates'][d]
        for s in range(prep['n_stocks']):
            val = result_matrix[d, s]
            if not np.isnan(val):
                results.append({
                    'trade_date': date,
                    'instrument': prep['unique_stocks'][s],
                    'high_low_diff': val
                })
    
    return pd.DataFrame(results)


# ========== 优化一：矩阵运算替代逐对循环 ==========

def spearman_corr_matrix(data: np.ndarray) -> np.ndarray:
    """
    批量计算Spearman相关系数矩阵。
    
    输入: data shape = (n_dates, n_stocks)，可能含NaN
    输出: corr_matrix shape = (n_stocks, n_stocks)
    
    原理：Spearman(X,Y) = Pearson(rank(X), rank(Y))
          对每列（每只股票）沿日期方向rank，然后计算Pearson相关矩阵
    """
    n_dates, n_stocks = data.shape
    if n_dates < 3 or n_stocks < 2:
        return np.full((n_stocks, n_stocks), np.nan)
    
    # Step 1: 每列rank（沿日期方向，axis=0）
    ranks = np.empty_like(data, dtype=float)
    for j in range(n_stocks):
        col = data[:, j]
        mask = ~np.isnan(col)
        if mask.sum() < 3:
            ranks[:, j] = np.nan
            continue
        r = np.empty_like(col, dtype=float)
        r[:] = np.nan
        r[mask] = stats.rankdata(col[mask])
        ranks[:, j] = r
    
    # Step 2: 中心化
    rank_means = np.nanmean(ranks, axis=0)
    rank_centered = ranks - rank_means
    
    # Step 3: 找到共同交易日
    common_mask = ~np.isnan(rank_centered).any(axis=1)
    n_common = common_mask.sum()
    
    if n_common < 3:
        return _fallback_pairwise_spearman(ranks)
    
    clean_ranks = rank_centered[common_mask, :]
    
    # Step 4: Pearson相关矩阵
    clean_means = np.mean(clean_ranks, axis=0)
    clean_centered = clean_ranks - clean_means
    cov_matrix = (clean_centered.T @ clean_centered) / (n_common - 1)
    
    stds = np.sqrt(np.diag(cov_matrix))
    stds[stds == 0] = np.nan
    
    corr_matrix = cov_matrix / np.outer(stds, stds)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    
    return corr_matrix


def _fallback_pairwise_spearman(ranks: np.ndarray) -> np.ndarray:
    """
    退化方案：逐对计算（当没有共同交易日时极少触发）
    """
    n_stocks = ranks.shape[1]
    corr_matrix = np.full((n_stocks, n_stocks), np.nan)
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            mask = ~np.isnan(ranks[:, i]) & ~np.isnan(ranks[:, j])
            if mask.sum() < 3:
                continue
            try:
                corr, _ = stats.spearmanr(ranks[mask, i], ranks[mask, j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
            except:
                pass
    np.fill_diagonal(corr_matrix, 1.0)
    return corr_matrix


def calculate_sui_bo_zhu_liu_vectorized(df_high_low_diff: pd.DataFrame) -> pd.DataFrame:
    """
    【优化一】向量化计算随波逐流因子。
    
    原方法：1000只股票 × 999对 × spearmanr()调用 = 百万次Python循环
    新方法：批量rank + 矩阵乘法 = 1次矩阵运算
    
    预期加速：10-50倍（取决于股票数量）
    """
    if df_high_low_diff.empty or len(df_high_low_diff) < MIN_STOCKS_FOR_CORR:
        return pd.DataFrame()
    
    # 构建透视表：行=日期，列=股票，值=high_low_diff
    pivot_df = df_high_low_diff.pivot(index='trade_date', columns='instrument', values='high_low_diff')
    if pivot_df.empty:
        return pd.DataFrame()
    
    all_dates = sorted(pivot_df.index)
    all_stocks = pivot_df.columns.tolist()
    n_dates = len(all_dates)
    n_stocks = len(all_stocks)
    
    print(f"随波逐流计算: {n_dates}个交易日, {n_stocks}只股票")
    print(f"使用向量化矩阵运算（无多进程）...")
    
    results = []
    
    for i in range(FACTOR_WINDOW - 1, n_dates):
        current_date = all_dates[i]
        window_dates = all_dates[i - FACTOR_WINDOW + 1:i + 1]
        window_data = pivot_df.loc[window_dates]
        
        # 筛选有效股票：窗口内至少80%数据非NaN
        valid_mask = window_data.count(axis=0) >= FACTOR_WINDOW * 0.8
        valid_stocks = window_data.columns[valid_mask].tolist()
        
        if len(valid_stocks) < MIN_STOCKS_FOR_CORR:
            continue
        
        # 提取子矩阵
        sub_matrix = window_data[valid_stocks].values  # shape: (n_dates_in_window, n_valid_stocks)
        
        # 【核心优化】矩阵运算计算Spearman相关矩阵
        corr_matrix = spearman_corr_matrix(sub_matrix)
        
        if corr_matrix is None or np.all(np.isnan(corr_matrix)):
            continue
        
        # 取绝对值，对角线置0（排除自身），求每行均值
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)
        
        # 计算每只股票的平均相关（与其他股票的绝对相关均值）
        avg_corr = np.nanmean(abs_corr, axis=1)
        
        # 生成结果
        for j, stock in enumerate(valid_stocks):
            if not np.isnan(avg_corr[j]):
                results.append({
                    'trade_date': current_date,
                    'instrument': stock,
                    'sui_bo_zhu_liu': float(avg_corr[j])
                })
        
        if (i - FACTOR_WINDOW + 2) % 50 == 0 or i == n_dates - 1:
            print(f"  进度: {i - FACTOR_WINDOW + 2}/{n_dates - FACTOR_WINDOW + 1} ({(i - FACTOR_WINDOW + 2)/(n_dates - FACTOR_WINDOW + 1)*100:.1f}%)")
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def calculate_gu_yan_chu_qun_vectorized(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    【优化二】向量化计算孤雁出群因子。
    
    原方法：逐对循环pearsonr()
    新方法：np.corrcoef()一次性计算Pearson矩阵
    
    预期加速：5-10倍
    """
    if df_minute.empty:
        return pd.DataFrame()
    
    df_minute = df_minute.sort_values(['instrument', 'date']).reset_index(drop=True)
    df_minute['minute_return'] = df_minute.groupby('instrument')['close'].pct_change()
    
    # 计算每分钟市场分化度（所有股票分钟收益率的标准差）
    # 【优化】先dropna()再走groupby std()的Cython fast path，避免Python lambda慢路径
    df_minute_clean = df_minute.dropna(subset=['minute_return'])
    all_minute_dates = df_minute['date'].unique()  # 保留完整日期列表用于reindex
    minute_divergence = (
        df_minute_clean.groupby('date')['minute_return']
        .std()
        .reindex(all_minute_dates)  # 保证空group日期出现为NaN
        .reset_index()
    )
    minute_divergence.columns = ['date', 'divergence']
    minute_divergence['trade_date'] = pd.to_datetime(minute_divergence['date']).dt.date
    
    # 每日均值分化度
    daily_mean_divergence = minute_divergence.groupby('trade_date')['divergence'].mean().reset_index()
    daily_mean_divergence.columns = ['trade_date', 'mean_divergence']
    minute_divergence = minute_divergence.merge(daily_mean_divergence, on='trade_date', how='left')
    minute_divergence['is_low_divergence'] = minute_divergence['divergence'] < minute_divergence['mean_divergence']
    
    # 筛选不分化时刻
    low_divergence_dates = minute_divergence[minute_divergence['is_low_divergence']]['date'].tolist()
    if not low_divergence_dates:
        return pd.DataFrame()
    
    df_low_div = df_minute[df_minute['date'].isin(low_divergence_dates)].copy()
    if df_low_div.empty:
        return pd.DataFrame()
    
    # 构建透视表：行=分钟时间戳，列=股票，值=amount
    pivot_df = df_low_div.pivot(index='date', columns='instrument', values='amount')
    pivot_df['trade_date'] = pd.to_datetime(pivot_df.index).date
    
    all_dates = sorted(pivot_df['trade_date'].unique())
    all_stocks = [c for c in pivot_df.columns if c != 'trade_date']
    
    print(f"孤雁出群计算: {len(all_dates)}个交易日, {len(all_stocks)}只股票")
    print(f"使用向量化矩阵运算（无多进程）...")
    
    results = []
    
    for trade_date in all_dates:
        # 提取当日数据
        day_data = pivot_df[pivot_df['trade_date'] == trade_date].drop(columns=['trade_date'])
        if day_data.empty:
            continue
        
        # 筛选有效股票（当日至少10个非NaN分钟数据）
        valid_mask = day_data.count(axis=0) >= 10
        valid_stocks = day_data.columns[valid_mask].tolist()
        
        if len(valid_stocks) < MIN_STOCKS_FOR_CORR:
            continue
        
        # 提取子矩阵
        sub_matrix = day_data[valid_stocks].values  # shape: (n_minutes, n_valid_stocks)
        
        # 【核心优化】np.corrcoef()一次性计算Pearson矩阵
        # rowvar=False: 每列是一个变量（每只股票），每行是一个观测（每分钟）
        corr_matrix = np.corrcoef(sub_matrix, rowvar=False)
        
        if np.all(np.isnan(corr_matrix)):
            continue
        
        # 取绝对值，对角线置0（排除自身），求每行均值
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)
        
        avg_corr = np.nanmean(abs_corr, axis=1)
        
        for j, stock in enumerate(valid_stocks):
            if not np.isnan(avg_corr[j]):
                results.append({
                    'trade_date': trade_date,
                    'instrument': stock,
                    'daily_gu_yan_chu_qun': float(avg_corr[j])
                })
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def calculate_daily_factors_vectorized(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    向量化计算日频随波逐流和孤雁出群因子（无多进程）。
    """
    print("获取日频数据用于计算合理收益率...")
    df_daily = fetch_daily_data_for_reasonable_return(start_date, end_date)
    if df_daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    print("计算合理收益率...")
    df_daily = calculate_reasonable_return(df_daily)
    
    print("获取流通市值数据...")
    market_cap_df = fetch_daily_market_cap(
        (pd.to_datetime(start_date) - pd.DateOffset(days=5)).strftime('%Y-%m-%d'),
        end_date
    )
    
    print("获取分钟级数据...")
    df_minute = fetch_minute_data_with_market(start_date, end_date)
    if df_minute.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    print("\n【优化四】计算高低额差（Numba日期级并行）...")
    prep = _encode_and_prepare_high_low(df_minute, df_daily, market_cap_df)
    print(f"  预处理完成: {prep['n_days']}个交易日, {prep['n_stocks']}只股票")
    print(f"  分钟数据总量: {len(prep['day_ids'])} 条")
    print(f"  启动Numba并行计算...")
    df_high_low_diff = calc_high_low_numba_wrapper(prep)
    print(f"  高低额差计算完成: {len(df_high_low_diff)} 条")
    
    if df_high_low_diff.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    print("\n【优化一】计算随波逐流因子（向量化矩阵运算）...")
    df_sui_bo = calculate_sui_bo_zhu_liu_vectorized(df_high_low_diff)
    
    print("\n【优化二】计算孤雁出群因子（向量化矩阵运算）...")
    df_gu_yan = calculate_gu_yan_chu_qun_vectorized(df_minute)
    
    return df_sui_bo, df_gu_yan


def calculate_shui_zhong_xing_zhou_factor(df_sui_bo: pd.DataFrame, df_gu_yan: pd.DataFrame) -> pd.DataFrame:
    """合成水中行舟因子（v3.3修正逻辑保留） """
    if df_sui_bo.empty and df_gu_yan.empty:
        return pd.DataFrame()
    
    if not df_gu_yan.empty:
        df_gu_yan = df_gu_yan.sort_values(['instrument', 'trade_date']).reset_index(drop=True)
        df_gu_yan['gu_yan_mean'] = df_gu_yan.groupby('instrument')['daily_gu_yan_chu_qun'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=10).mean()
        )
        df_gu_yan['gu_yan_std'] = df_gu_yan.groupby('instrument')['daily_gu_yan_chu_qun'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=10).std()
        )
        df_gu_yan['gu_yan_chu_qun_20'] = (df_gu_yan['gu_yan_mean'] + df_gu_yan['gu_yan_std']) / 2
    
    if df_sui_bo.empty:
        df_merged = df_gu_yan.rename(columns={'trade_date': 'date'})
        df_merged['shui_zhong_xing_zhou'] = df_merged['gu_yan_chu_qun_20']
    elif df_gu_yan.empty:
        df_merged = df_sui_bo.rename(columns={'trade_date': 'date'})
        df_merged['shui_zhong_xing_zhou'] = df_merged['sui_bo_zhu_liu']
    else:
        df_merged = df_sui_bo.merge(
            df_gu_yan[['trade_date', 'instrument', 'gu_yan_chu_qun_20']],
            on=['trade_date', 'instrument'],
            how='outer'
        )
        df_merged['date'] = pd.to_datetime(df_merged['trade_date'])
        
        for col in ['sui_bo_zhu_liu', 'gu_yan_chu_qun_20']:
            if col in df_merged.columns:
                df_merged[col] = df_merged.groupby('trade_date')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        df_merged['shui_zhong_xing_zhou'] = (
            df_merged['sui_bo_zhu_liu'] + df_merged['gu_yan_chu_qun_20']
        ) / 2
    
    return df_merged[['date', 'instrument', 'shui_zhong_xing_zhou']].dropna()


# ========== 主流程 ==========
if __name__ == "__main__":
    import time
    start_time_total = time.time()
    
    start_date = "2026-02-04"
    end_date = "2026-03-14"
    overwrite = False
    use_incremental = True
    
    print(f"=== {FACTOR_NAME}（水中行舟）因子计算 - v3.5 Numba加速版 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print()
    print("【v3.5优化内容】")
    print("1. 高低额差：逐日Pandas循环 → Numba日期级并行（prange），预期加速5-15倍")
    print("2. 随波逐流：逐对循环spearmanr → 矩阵运算（批量rank + 矩阵乘法），加速10-50倍")
    print("3. 孤雁出群：逐对循环pearsonr → np.corrcoef矩阵运算，加速5-10倍")
    print("4. 移除ProcessPoolExecutor多进程，避免大数据序列化开销")
    print("5. 保留v3.3所有修正逻辑（日期对齐、无双重平滑、截面均值填充）")
    print()
    
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            effective_start = last_computed
            print(f"【增量模式】实际计算起始: {effective_start}")
        else:
            effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=40)).strftime('%Y-%m-%d')
            print(f"【全量模式】需要历史数据，扩展起始: {effective_start}")
    else:
        effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=40)).strftime('%Y-%m-%d')
        print(f"【全量模式】扩展起始: {effective_start}")
    
    print("\n【步骤1】向量化计算日频随波逐流和孤雁出群因子...")
    start_time_calc = time.time()
    df_sui_bo, df_gu_yan = calculate_daily_factors_vectorized(effective_start, end_date)
    calc_time = time.time() - start_time_calc
    
    if df_sui_bo.empty and df_gu_yan.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"\n日频因子计算完成（耗时 {calc_time:.2f}秒）:")
        print(f"  随波逐流: {len(df_sui_bo)} 条" if not df_sui_bo.empty else "  随波逐流: 无数据")
        print(f"  孤雁出群: {len(df_gu_yan)} 条" if not df_gu_yan.empty else "  孤雁出群: 无数据")
        
        print("\n【步骤2】合成水中行舟因子...")
        df_factor = calculate_shui_zhong_xing_zhou_factor(df_sui_bo, df_gu_yan)
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
        else:
            print(f"因子计算完成: {len(df_factor)} 条")
            
            df_factor['date'] = pd.to_datetime(df_factor['date'])
            original_count = len(df_factor)
            df_factor = df_factor[
                (df_factor['date'] >= pd.to_datetime(start_date)) &
                (df_factor['date'] <= pd.to_datetime(end_date))
            ].reset_index(drop=True)
            
            print(f"\n过滤到指定日期范围: {original_count} -> {len(df_factor)} 条")
            print(f"日期范围: {df_factor['date'].min().date()} ~ {df_factor['date'].max().date()}")
            
            if not df_factor.empty:
                df_to_write = prepare_factor_df_for_write(df_factor, FACTOR_NAME)
                start_time_write = time.time()
                written_count = safe_write_to_library(
                    df_to_write, table_id=FACTOR_LIBRARY_TABLE,
                    factor_name=FACTOR_NAME, overwrite=overwrite
                )
                write_time = time.time() - start_time_write
                
                total_time = time.time() - start_time_total
                print(f"\n=== 完成 ===")
                print(f"计算耗时: {calc_time:.2f}秒")
                print(f"写入耗时: {write_time:.2f}秒")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"'{FACTOR_NAME}' 共写入 {written_count} 条数据到 '{FACTOR_LIBRARY_TABLE}'（宽表）")
                print(f"\n预期表现: Rank IC ~ -9.36%（研报基准）")
            else:
                print("过滤后没有数据需要写入。")

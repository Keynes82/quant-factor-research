# A09 水中行舟因子 - v3.3 全量修正版（daikin01）
# 来源：方正证券《个股成交额的市场跟随性与"水中行舟"因子》（2023-02-15）
# 档案编号：FQ-20260403-009
# 因子名称：A09（水中行舟）
# 版本：v3.3 - 修复核心BUG
#
# 【v3.3修正内容】
#   1. 修复随波逐流相关系数日期对齐BUG（不同股票缺失日期不同时，
#      原代码用[:min_len]切片会导致日期错位，Spearman相关系数完全失真）
#   2. 移除随波逐流双重平滑（sui_bo_zhu_liu本身已是20日窗口结果，
#      合成阶段不应再做rolling(20)）
#   3. 修复缺失值0填充BUG（0填充会扭曲因子分布，改为截面均值填充）
#   4. 保留v3.2多核并行性能优化
#
# 因子构建逻辑（严格按研报定义）：
#   【随波逐流因子】- 正向因子（股价高位时，成交额跟随市场越好）
#   1. 计算合理收益率 = 过去20日日内收益率（收盘/开盘-1）的均值
#   2. 分钟相对开盘收益率 = 分钟收盘价 / 当日开盘 - 1
#   3. 高位成交额 = 相对开盘收益率 > 合理收益率 的分钟成交额之和
#   4. 低位成交额 = 相对开盘收益率 < 合理收益率 的分钟成交额之和
#   5. 高低额差 = (高位成交额 - 低位成交额) / 流通市值
#   6. 随波逐流 = 每只股票与其他股票过去20日"高低额差"序列的spearman相关系数绝对值的均值
#
#   【孤雁出群因子】- 负向因子（市场不分化时，成交额独立越好）
#   1. 分钟市场分化度 = 每分钟所有股票分钟收益率的标准差
#   2. 不分化时刻 = 分钟市场分化度 < 当日均值 的时刻
#   3. 日孤雁出群 = 每只股票在"不分化时刻"的分钟成交额与其他股票分钟成交额的pearson相关系数绝对值的均值
#   4. 孤雁出群 = (20日均值 + 20日标准差) / 2
#
#   【合成】
#   水中行舟因子 = (随波逐流 + 孤雁出群) / 2
#
# 数据表：
#   - cn_stock_bar1m: 分钟级数据（open, high, low, close, volume, amount）
#   - cn_stock_bar1d: 日频数据（open, close, 用于计算日内收益率）
#   - cn_stock_prefactors: 因子数据表（float_market_cap 流通市值）

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A09"                # 本因子编号（水中行舟）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数（增量计算时往前推）
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
REASONABLE_RETURN_WINDOW = 20      # 合理收益率计算窗口（20日）
MIN_STOCKS_FOR_CORR = 50           # 计算相关性所需的最小股票数

# ========== 性能配置 ==========
MAX_WORKERS = 8                    # 并行进程数
BATCH_SIZE = 100                   # 每批处理的股票数量


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


def safe_write_to_library(df: pd.DataFrame, 
                          table_id: str = FACTOR_LIBRARY_TABLE, 
                          factor_name: str = FACTOR_NAME, 
                          overwrite: bool = False) -> int:
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
        print(f"成功写入 '{factor_name}'：{len(df)} 行到 '{table_id}'")
        return len(df)
        
    except Exception as e:
        print(f"写入失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def chunk_list(lst: List, size: int):
    """列表分块生成器"""
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def fetch_daily_data_for_reasonable_return(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取日频数据用于计算合理收益率。
    合理收益率 = 过去20日日内收益率（收盘/开盘-1）的均值
    """
    start_dt = (pd.to_datetime(start_date) - pd.DateOffset(days=REASONABLE_RETURN_WINDOW + 5)).strftime('%Y-%m-%d')
    
    sql = f"""
    SELECT 
        CAST(date AS DATE) AS trade_date,
        instrument,
        open,
        close
    FROM cn_stock_bar1d
    WHERE date >= '{start_dt}'
      AND date <= '{end_date}'
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
    """
    计算合理收益率 = 过去20日日内收益率的均值
    """
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'trade_date']).reset_index(drop=True)
    
    df['reasonable_return'] = df.groupby('instrument')['intraday_return'].transform(
        lambda x: x.rolling(window=REASONABLE_RETURN_WINDOW, min_periods=10).mean()
    )
    
    return df


def fetch_minute_data_with_market(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取分钟级数据，包含计算市场分化度所需的所有股票数据。
    数据时间过滤：09:30-15:00（剔除集合竞价）
    """
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
    """
    获取流通市值数据。
    使用 cn_stock_prefactors 表，字段 float_market_cap（流通市值）
    """
    sql = f"""
    SELECT 
        CAST(date AS DATE) AS trade_date,
        instrument,
        float_market_cap AS circulating_market_cap
    FROM cn_stock_prefactors
    WHERE date >= '{start_date}'
      AND date <= '{end_date}'
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


def calculate_daily_high_low_amount(df_minute: pd.DataFrame, reasonable_return_df: pd.DataFrame, 
                                    market_cap_df: pd.DataFrame, trade_date) -> pd.DataFrame:
    """
    计算单日的高低额差。
    """
    if df_minute.empty:
        return pd.DataFrame()
    
    reasonable_return_today = reasonable_return_df[
        reasonable_return_df['trade_date'] == trade_date
    ][['instrument', 'reasonable_return']].copy()
    
    if reasonable_return_today.empty:
        return pd.DataFrame()
    
    daily_open = df_minute.groupby('instrument')['open'].first().reset_index()
    daily_open.columns = ['instrument', 'daily_open']
    
    df_minute = df_minute.merge(daily_open, on='instrument', how='left')
    df_minute['relative_open_return'] = df_minute['close'] / df_minute['daily_open'] - 1
    
    df_minute = df_minute.merge(reasonable_return_today, on='instrument', how='left')
    
    df_minute['high_amount'] = np.where(
        df_minute['relative_open_return'] > df_minute['reasonable_return'],
        df_minute['amount'],
        0
    )
    df_minute['low_amount'] = np.where(
        df_minute['relative_open_return'] < df_minute['reasonable_return'],
        df_minute['amount'],
        0
    )
    
    daily_summary = df_minute.groupby('instrument').agg({
        'high_amount': 'sum',
        'low_amount': 'sum'
    }).reset_index()
    
    market_cap_today = market_cap_df[market_cap_df['trade_date'] == trade_date]
    daily_summary = daily_summary.merge(
        market_cap_today[['instrument', 'circulating_market_cap']],
        on='instrument',
        how='left'
    )
    
    daily_summary['high_low_diff'] = (
        daily_summary['high_amount'] - daily_summary['low_amount']
    ) / daily_summary['circulating_market_cap']
    
    daily_summary['trade_date'] = trade_date
    
    return daily_summary[['trade_date', 'instrument', 'high_low_diff']].dropna()


# ========== 高性能并行计算核心（v3.3修正版） ==========

def calculate_spearman_correlations_single_date(args):
    """
    并行计算单日的随波逐流因子（v3.3修正版）。
    
    【关键修正】v3.2中 stock_series.dropna().values 后直接用[:min_len]切片对齐，
    但不同股票缺失日期不同时会导致日期错位。v3.3改用 index.intersection 精确对齐。
    
    参数:
        args = (current_date, window_dates, pivot_data_dict, valid_stocks_list)
    """
    current_date, window_dates, pivot_data_dict, valid_stocks_list = args
    
    window_data = pd.DataFrame(pivot_data_dict)
    if window_data.empty:
        return []
    
    results = []
    
    for stock in valid_stocks_list:
        if stock not in window_data.columns:
            continue
        
        # 【v3.3修正】保留Series的index，用日期对齐，不取.values丢失index
        stock_series = window_data[stock].dropna()
        
        # 检查序列有效性
        if len(stock_series) < FACTOR_WINDOW * 0.8 or len(set(stock_series.values)) <= 1:
            continue
        
        correlations = []
        
        for other_stock in valid_stocks_list:
            if other_stock == stock or other_stock not in window_data.columns:
                continue
            
            other_series = window_data[other_stock].dropna()
            
            if len(other_series) < FACTOR_WINDOW * 0.8 or len(set(other_series.values)) <= 1:
                continue
            
            # 【v3.3核心修正】用日期index精确对齐，而非切片
            common_dates = stock_series.index.intersection(other_series.index)
            if len(common_dates) < 10:
                continue
            
            try:
                s1 = stock_series.loc[common_dates].values
                s2 = other_series.loc[common_dates].values
                corr, _ = stats.spearmanr(s1, s2)
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            except:
                continue
        
        if correlations:
            avg_corr = np.mean(correlations)
            results.append({
                'trade_date': current_date,
                'instrument': stock,
                'sui_bo_zhu_liu': avg_corr
            })
    
    return results


def calculate_pearson_correlations_single_date(args):
    """
    并行计算单日的孤雁出群因子（v3.1逻辑，此部分无BUG）。
    
    参数:
        args = (trade_date, minute_pivot_dict, valid_stocks_list)
    """
    trade_date, minute_pivot_dict, valid_stocks_list = args
    
    day_data = pd.DataFrame(minute_pivot_dict)
    if day_data.empty:
        return []
    
    results = []
    
    for stock in valid_stocks_list:
        if stock not in day_data.columns:
            continue
        
        stock_series = day_data[stock].dropna()
        if len(stock_series) < 10 or len(set(stock_series.values)) <= 1:
            continue
        
        stock_values = stock_series.values
        correlations = []
        
        for other_stock in valid_stocks_list:
            if other_stock == stock or other_stock not in day_data.columns:
                continue
            
            other_series = day_data[other_stock].dropna()
            if len(other_series) < 10 or len(set(other_series.values)) <= 1:
                continue
            
            common_index = stock_series.index.intersection(other_series.index)
            if len(common_index) < 5:
                continue
            
            try:
                corr, _ = stats.pearsonr(
                    stock_series.loc[common_index].values,
                    other_series.loc[common_index].values
                )
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            except:
                continue
        
        if correlations:
            avg_corr = np.mean(correlations)
            results.append({
                'trade_date': trade_date,
                'instrument': stock,
                'daily_gu_yan_chu_qun': avg_corr
            })
    
    return results


def calculate_sui_bo_zhu_liu_parallel(df_high_low_diff: pd.DataFrame, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    """
    高性能并行计算随波逐流因子。
    """
    if df_high_low_diff.empty or len(df_high_low_diff) < MIN_STOCKS_FOR_CORR:
        return pd.DataFrame()
    
    pivot_df = df_high_low_diff.pivot(index='trade_date', columns='instrument', values='high_low_diff')
    if pivot_df.empty:
        return pd.DataFrame()
    
    all_dates = sorted(pivot_df.index)
    all_stocks = pivot_df.columns.tolist()
    
    print(f"随波逐流计算: {len(all_dates)}个交易日, {len(all_stocks)}只股票")
    print(f"使用{max_workers}核并行计算...")
    
    task_args = []
    for i in range(FACTOR_WINDOW - 1, len(all_dates)):
        current_date = all_dates[i]
        window_dates = all_dates[i - FACTOR_WINDOW + 1:i + 1]
        window_data = pivot_df.loc[window_dates]
        
        valid_stocks = []
        for stock in all_stocks:
            series = window_data[stock].dropna()
            if len(series) >= FACTOR_WINDOW * 0.8:
                valid_stocks.append(stock)
        
        if len(valid_stocks) < MIN_STOCKS_FOR_CORR:
            continue
        
        pivot_data_dict = window_data.to_dict()
        task_args.append((current_date, window_dates, pivot_data_dict, valid_stocks))
    
    if not task_args:
        return pd.DataFrame()
    
    all_results = []
    completed = 0
    total = len(task_args)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calculate_spearman_correlations_single_date, arg): arg for arg in task_args}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"  进度: {completed}/{total} ({completed/total*100:.1f}%)")
            except Exception as e:
                print(f"  计算错误: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.DataFrame(all_results)


def calculate_gu_yan_chu_qun_parallel(df_minute: pd.DataFrame, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    """
    高性能并行计算孤雁出群因子。
    """
    if df_minute.empty:
        return pd.DataFrame()
    
    df_minute = df_minute.sort_values(['instrument', 'date']).reset_index(drop=True)
    df_minute['minute_return'] = df_minute.groupby('instrument')['close'].pct_change()
    
    minute_divergence = df_minute.groupby('date').agg({
        'minute_return': lambda x: x.dropna().std(),
    }).reset_index()
    minute_divergence.columns = ['date', 'divergence']
    minute_divergence['trade_date'] = pd.to_datetime(minute_divergence['date']).dt.date
    
    daily_mean_divergence = minute_divergence.groupby('trade_date')['divergence'].mean().reset_index()
    daily_mean_divergence.columns = ['trade_date', 'mean_divergence']
    minute_divergence = minute_divergence.merge(daily_mean_divergence, on='trade_date', how='left')
    minute_divergence['is_low_divergence'] = minute_divergence['divergence'] < minute_divergence['mean_divergence']
    
    low_divergence_minutes = minute_divergence[minute_divergence['is_low_divergence']]['date'].tolist()
    if not low_divergence_minutes:
        return pd.DataFrame()
    
    df_low_div = df_minute[df_minute['date'].isin(low_divergence_minutes)].copy()
    if df_low_div.empty:
        return pd.DataFrame()
    
    pivot_df = df_low_div.pivot(index='date', columns='instrument', values='amount')
    pivot_df['trade_date'] = pd.to_datetime(pivot_df.index).date
    
    all_dates = sorted(pivot_df['trade_date'].unique())
    all_stocks = [c for c in pivot_df.columns if c != 'trade_date']
    
    print(f"孤雁出群计算: {len(all_dates)}个交易日, {len(all_stocks)}只股票")
    print(f"使用{max_workers}核并行计算...")
    
    task_args = []
    for trade_date in all_dates:
        day_data = pivot_df[pivot_df['trade_date'] == trade_date].drop(columns=['trade_date'])
        if day_data.empty:
            continue
        
        valid_stocks = []
        for stock in day_data.columns:
            if len(day_data[stock].dropna()) >= 10:
                valid_stocks.append(stock)
        
        if len(valid_stocks) < MIN_STOCKS_FOR_CORR:
            continue
        
        minute_pivot_dict = day_data.to_dict()
        task_args.append((trade_date, minute_pivot_dict, valid_stocks))
    
    if not task_args:
        return pd.DataFrame()
    
    all_results = []
    completed = 0
    total = len(task_args)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calculate_pearson_correlations_single_date, arg): arg for arg in task_args}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"  进度: {completed}/{total} ({completed/total*100:.1f}%)")
            except Exception as e:
                print(f"  计算错误: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.DataFrame(all_results)


def calculate_daily_factors_parallel(start_date: str, end_date: str, max_workers: int = MAX_WORKERS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    并行计算日频随波逐流和孤雁出群因子。
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
    
    print("\n计算随波逐流因子（并行）...")
    all_high_low_diff = []
    for trade_date in df_minute['trade_date'].unique():
        day_minute = df_minute[df_minute['trade_date'] == trade_date]
        day_high_low = calculate_daily_high_low_amount(day_minute, df_daily, market_cap_df, trade_date)
        if not day_high_low.empty:
            all_high_low_diff.append(day_high_low)
    
    if not all_high_low_diff:
        return pd.DataFrame(), pd.DataFrame()
    
    df_high_low_diff = pd.concat(all_high_low_diff, ignore_index=True)
    
    df_sui_bo = calculate_sui_bo_zhu_liu_parallel(df_high_low_diff, max_workers=max_workers)
    
    print("\n计算孤雁出群因子（并行）...")
    df_gu_yan = calculate_gu_yan_chu_qun_parallel(df_minute, max_workers=max_workers)
    
    return df_sui_bo, df_gu_yan


def calculate_shui_zhong_xing_zhou_factor(df_sui_bo: pd.DataFrame, df_gu_yan: pd.DataFrame) -> pd.DataFrame:
    """
    合成水中行舟因子（v3.3修正版）。
    
    【v3.3修正】
    1. 随波逐流因子(sui_bo_zhu_liu)本身已是20日窗口结果，不再做二次rolling平滑
    2. 缺失值用截面均值填充，而非0填充
    """
    if df_sui_bo.empty and df_gu_yan.empty:
        return pd.DataFrame()
    
    # 孤雁出群因子：20日rolling（均值+标准差）/2 —— 研报定义，保留
    if not df_gu_yan.empty:
        df_gu_yan = df_gu_yan.sort_values(['instrument', 'trade_date']).reset_index(drop=True)
        df_gu_yan['gu_yan_mean'] = df_gu_yan.groupby('instrument')['daily_gu_yan_chu_qun'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=10).mean()
        )
        df_gu_yan['gu_yan_std'] = df_gu_yan.groupby('instrument')['daily_gu_yan_chu_qun'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=10).std()
        )
        df_gu_yan['gu_yan_chu_qun_20'] = (df_gu_yan['gu_yan_mean'] + df_gu_yan['gu_yan_std']) / 2
    
    # 合并两个因子
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
        
        # 【v3.3修正】用截面均值填充缺失值，不扭曲因子排序
        for col in ['sui_bo_zhu_liu', 'gu_yan_chu_qun_20']:
            if col in df_merged.columns:
                # 按日期截面计算均值填充
                df_merged[col] = df_merged.groupby('trade_date')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        # 合成水中行舟因子 = (随波逐流 + 孤雁出群) / 2
        df_merged['shui_zhong_xing_zhou'] = (
            df_merged['sui_bo_zhu_liu'] + df_merged['gu_yan_chu_qun_20']
        ) / 2
    
    return df_merged[['date', 'instrument', 'shui_zhong_xing_zhou']].dropna()


# ========== 主流程 ==========
if __name__ == "__main__":
    import time
    start_time_total = time.time()
    
    # 参数配置
    start_date = "2026-02-04"
    end_date = "2026-03-14"
    overwrite = False
    use_incremental = True
    max_workers = MAX_WORKERS
    
    print(f"=== {FACTOR_NAME}（水中行舟）因子计算 - v3.3全量修正版 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"并行核数: {max_workers}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print()
    print("【v3.3修正内容】")
    print("1. 修复随波逐流相关系数日期对齐BUG")
    print("2. 移除随波逐流双重平滑")
    print("3. 修复缺失值0填充BUG（改用截面均值填充）")
    print()
    
    # 增量计算
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
    
    # 步骤1：并行计算日频因子
    print("\n【步骤1】并行计算日频随波逐流和孤雁出群因子...")
    start_time_calc = time.time()
    df_sui_bo, df_gu_yan = calculate_daily_factors_parallel(effective_start, end_date, max_workers=max_workers)
    calc_time = time.time() - start_time_calc
    
    if df_sui_bo.empty and df_gu_yan.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"\n日频因子计算完成（耗时 {calc_time:.2f}秒）:")
        print(f"  随波逐流: {len(df_sui_bo)} 条" if not df_sui_bo.empty else "  随波逐流: 无数据")
        print(f"  孤雁出群: {len(df_gu_yan)} 条" if not df_gu_yan.empty else "  孤雁出群: 无数据")
        
        # 步骤2：合成水中行舟因子
        print("\n【步骤2】合成水中行舟因子...")
        df_factor = calculate_shui_zhong_xing_zhou_factor(df_sui_bo, df_gu_yan)
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
        else:
            print(f"因子计算完成: {len(df_factor)} 条")
            
            # 过滤到用户指定日期范围
            df_factor['date'] = pd.to_datetime(df_factor['date'])
            original_count = len(df_factor)
            df_factor = df_factor[
                (df_factor['date'] >= pd.to_datetime(start_date)) & 
                (df_factor['date'] <= pd.to_datetime(end_date))
            ].reset_index(drop=True)
            
            print(f"\n过滤到指定日期范围: {original_count} -> {len(df_factor)} 条")
            print(f"日期范围: {df_factor['date'].min().date()} ~ {df_factor['date'].max().date()}")
            
            if not df_factor.empty:
                print(f"日频数据: {len(df_factor)}条")
                
                # 准备宽表格式数据
                df_to_write = prepare_factor_df_for_write(df_factor, FACTOR_NAME)
                
                # 写入宽表
                start_time_write = time.time()
                written_count = safe_write_to_library(
                    df_to_write,
                    table_id=FACTOR_LIBRARY_TABLE,
                    factor_name=FACTOR_NAME,
                    overwrite=overwrite
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

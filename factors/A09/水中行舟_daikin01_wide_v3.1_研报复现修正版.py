# A09 水中行舟因子 - 宽表版本（daikin01）
# 来源：方正证券《个股成交额的市场跟随性与"水中行舟"因子》（2023-02-15）
# 档案编号：FQ-20260403-009
# 因子名称：A09（水中行舟）
# 版本：v3.1 - 修正相关系数计算（20日时间序列而非单日截面）
#
# 修正说明（v3.0 -> v3.1）：
#   【随波逐流因子】
#   - v3.0问题：使用单日截面数据计算相关系数（常数序列导致统计错误）
#   - v3.1修正：使用过去20个交易日的高低额差时间序列计算spearman相关系数
#   
#   【孤雁出群因子】
#   - v3.0问题：使用单日截面数据计算相关系数
#   - v3.1修正：使用当日"不分化时刻"的分钟成交额时间序列计算pearson相关系数
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
#   - cn_stock_daily: 日频数据（流通市值 circulating_market_cap）

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Dict
from scipy import stats

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A09"                # 本因子编号（水中行舟）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数（增量计算时往前推）
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
REASONABLE_RETURN_WINDOW = 20      # 合理收益率计算窗口（20日）
MIN_STOCKS_FOR_CORR = 50           # 计算相关性所需的最小股票数


def get_last_computed_date(table_id: str, factor_name: str = FACTOR_NAME) -> str:
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


def fetch_all_a_share_instruments() -> List[str]:
    """从平台读取所有 A 股的 instrument 列表。"""
    sql = "SELECT DISTINCT instrument FROM cn_stock_instruments"
    q = dai.query(sql, full_db_scan=True)
    df = q.df()
    if df is None or df.empty:
        return []
    return df['instrument'].astype(str).tolist()


def fetch_daily_data_for_reasonable_return(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取日频数据用于计算合理收益率。
    合理收益率 = 过去20日日内收益率（收盘/开盘-1）的均值
    """
    # 需要额外获取20日历史数据
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
        
        # 计算日内收益率
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
    
    # 20日滚动平均
    df['reasonable_return'] = df.groupby('instrument')['intraday_return'].transform(
        lambda x: x.rolling(window=REASONABLE_RETURN_WINDOW, min_periods=10).mean()
    )
    
    return df


def fetch_minute_data_with_market(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取分钟级数据，包含计算市场分化度所需的所有股票数据。
    数据时间过滤：09:35-14:56（剔除开盘前5分钟和收盘前3分钟）
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
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) >= 935
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) <= 1456
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
    """
    sql = f"""
    SELECT 
        CAST(date AS DATE) AS trade_date,
        instrument,
        circulating_market_cap
    FROM cn_stock_daily
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
                                    market_cap_df: pd.DataFrame, trade_date: datetime.date) -> pd.DataFrame:
    """
    计算单日的高低额差。
    
    步骤：
    1. 获取当日开盘价（用于计算相对开盘收益率）
    2. 计算分钟相对开盘收益率
    3. 根据合理收益率阈值，分离高位成交额和低位成交额
    4. 计算高低额差 = (高位 - 低位) / 流通市值
    """
    if df_minute.empty:
        return pd.DataFrame()
    
    # 获取当日的合理收益率
    reasonable_return_today = reasonable_return_df[
        reasonable_return_df['trade_date'] == trade_date
    ][['instrument', 'reasonable_return']].copy()
    
    if reasonable_return_today.empty:
        return pd.DataFrame()
    
    # 获取当日开盘价格
    daily_open = df_minute.groupby('instrument')['open'].first().reset_index()
    daily_open.columns = ['instrument', 'daily_open']
    
    # 计算每只股票的分钟相对开盘收益率
    df_minute = df_minute.merge(daily_open, on='instrument', how='left')
    df_minute['relative_open_return'] = df_minute['close'] / df_minute['daily_open'] - 1
    
    # 合并合理收益率
    df_minute = df_minute.merge(reasonable_return_today, on='instrument', how='left')
    
    # 分离高位和低位成交额
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
    
    # 按股票汇总
    daily_summary = df_minute.groupby('instrument').agg({
        'high_amount': 'sum',
        'low_amount': 'sum'
    }).reset_index()
    
    # 合并流通市值
    market_cap_today = market_cap_df[market_cap_df['trade_date'] == trade_date]
    daily_summary = daily_summary.merge(
        market_cap_today[['instrument', 'circulating_market_cap']],
        on='instrument',
        how='left'
    )
    
    # 计算高低额差
    daily_summary['high_low_diff'] = (
        daily_summary['high_amount'] - daily_summary['low_amount']
    ) / daily_summary['circulating_market_cap']
    
    daily_summary['trade_date'] = trade_date
    
    return daily_summary[['trade_date', 'instrument', 'high_low_diff']].dropna()


def calculate_sui_bo_zhu_liu(df_high_low_diff: pd.DataFrame) -> pd.DataFrame:
    """
    计算随波逐流因子（修正版）。
    
    研报原文：
    "每月月底，取所有股票过去20个交易日的'高低额差'序列。
    分别计算每只股票与其余所有股票的'高低额差'序列之间的spearman相关系数"
    
    修正要点：
    - 使用过去20个交易日的时间序列数据（而非单日截面）
    - 计算两只股票20日高低额差序列的spearman相关系数
    - 取绝对值后求均值
    """
    if df_high_low_diff.empty or len(df_high_low_diff) < MIN_STOCKS_FOR_CORR:
        return pd.DataFrame()
    
    # 构建透视表：行为日期，列为股票，值为高低额差
    pivot_df = df_high_low_diff.pivot(index='trade_date', columns='instrument', values='high_low_diff')
    
    if pivot_df.empty:
        return pd.DataFrame()
    
    # 按股票分组，计算20日滚动窗口内的相关系数
    results = []
    
    # 获取所有交易日和股票
    all_dates = sorted(pivot_df.index)
    all_stocks = pivot_df.columns.tolist()
    
    # 从第20个交易日开始计算（需要足够的历史数据）
    for i in range(FACTOR_WINDOW - 1, len(all_dates)):
        current_date = all_dates[i]
        # 获取过去20个交易日的数据
        window_dates = all_dates[i - FACTOR_WINDOW + 1:i + 1]
        window_data = pivot_df.loc[window_dates]
        
        # 过滤掉有太多缺失值的股票
        valid_stocks = []
        for stock in all_stocks:
            series = window_data[stock].dropna()
            if len(series) >= FACTOR_WINDOW * 0.8:  # 至少80%的数据存在
                valid_stocks.append(stock)
        
        if len(valid_stocks) < MIN_STOCKS_FOR_CORR:
            continue
        
        # 对每只股票，计算它与其他股票的spearman相关系数
        for stock in valid_stocks:
            correlations = []
            stock_series = window_data[stock].values
            
            # 检查当前股票序列是否有变化
            if len(set(stock_series)) <= 1:
                continue
            
            for other_stock in valid_stocks:
                if other_stock == stock:
                    continue
                
                other_series = window_data[other_stock].values
                
                # 检查其他股票序列是否有变化
                if len(set(other_series)) <= 1:
                    continue
                
                # 计算两个20日序列的spearman相关系数
                try:
                    corr, _ = stats.spearmanr(stock_series, other_series)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    continue
            
            if correlations:
                # 取所有相关系数绝对值的均值
                avg_corr = np.mean(correlations)
                results.append({
                    'trade_date': current_date,
                    'instrument': stock,
                    'sui_bo_zhu_liu': avg_corr
                })
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def calculate_gu_yan_chu_qun(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算孤雁出群因子（修正版）。
    
    研报原文：
    "取市场上所有股票在当日'不分化时刻'的成交额序列，
    然后分别计算每只股票的分钟成交额序列与其余股票分钟成交额序列的pearson相关系数"
    
    修正要点：
    - 使用分钟级时间序列数据计算相关系数（而非日频单值）
    - 对每只股票，获取其在当日所有"不分化时刻"的成交额序列
    - 计算两只股票分钟成交额序列的pearson相关系数
    - 然后20日低频化：(20日均值 + 20日标准差) / 2
    """
    if df_minute.empty:
        return pd.DataFrame()
    
    # 计算分钟收益率
    df_minute = df_minute.sort_values(['instrument', 'date']).reset_index(drop=True)
    df_minute['minute_return'] = df_minute.groupby('instrument')['close'].pct_change()
    
    # 计算分钟市场分化度（每分钟所有股票分钟收益率的标准差）
    minute_divergence = df_minute.groupby('date').agg({
        'minute_return': lambda x: x.dropna().std(),
        'amount': 'sum'  # 市场总成交额
    }).reset_index()
    minute_divergence.columns = ['date', 'divergence', 'market_amount']
    
    # 计算当日市场分化度均值
    minute_divergence['trade_date'] = pd.to_datetime(minute_divergence['date']).dt.date
    daily_mean_divergence = minute_divergence.groupby('trade_date')['divergence'].mean().reset_index()
    daily_mean_divergence.columns = ['trade_date', 'mean_divergence']
    
    # 识别不分化时刻
    minute_divergence = minute_divergence.merge(daily_mean_divergence, on='trade_date', how='left')
    minute_divergence['is_low_divergence'] = minute_divergence['divergence'] < minute_divergence['mean_divergence']
    
    # 只保留不分化时刻
    low_divergence_minutes = minute_divergence[minute_divergence['is_low_divergence']]['date'].tolist()
    
    if not low_divergence_minutes:
        return pd.DataFrame()
    
    # 筛选不分化时刻的数据
    df_low_div = df_minute[df_minute['date'].isin(low_divergence_minutes)].copy()
    
    if df_low_div.empty:
        return pd.DataFrame()
    
    # 构建透视表：行为分钟时间戳，列为股票，值为成交额
    # 这样每只股票有一个分钟级成交额序列
    pivot_df = df_low_div.pivot(index='date', columns='instrument', values='amount')
    pivot_df['trade_date'] = pd.to_datetime(pivot_df.index).date
    
    # 按交易日分组，计算每日的"日孤雁出群"
    daily_results = []
    
    for trade_date in pivot_df['trade_date'].unique():
        # 获取该日的分钟数据（去掉trade_date列）
        day_data = pivot_df[pivot_df['trade_date'] == trade_date].drop(columns=['trade_date'])
        
        if day_data.empty:
            continue
        
        # 过滤掉有太多缺失值的股票（至少要有10个有效分钟数据）
        valid_stocks = []
        for stock in day_data.columns:
            series = day_data[stock].dropna()
            if len(series) >= 10:  # 至少10个分钟数据点
                valid_stocks.append(stock)
        
        if len(valid_stocks) < MIN_STOCKS_FOR_CORR:
            continue
        
        # 对每只股票，计算它与其他股票的pearson相关系数
        for stock in valid_stocks:
            correlations = []
            stock_series = day_data[stock].dropna().values
            
            # 检查当前股票序列是否有变化
            if len(set(stock_series)) <= 1:
                continue
            
            for other_stock in valid_stocks:
                if other_stock == stock:
                    continue
                
                other_series = day_data[other_stock].dropna().values
                
                # 检查其他股票序列是否有变化
                if len(set(other_series)) <= 1:
                    continue
                
                # 对齐两个序列（取交集的索引）
                # 使用pandas的索引对齐
                aligned_stock = day_data[stock].dropna()
                aligned_other = day_data[other_stock].dropna()
                common_index = aligned_stock.index.intersection(aligned_other.index)
                
                if len(common_index) < 5:  # 至少需要5个共同时间点
                    continue
                
                stock_aligned = aligned_stock.loc[common_index].values
                other_aligned = aligned_other.loc[common_index].values
                
                # 计算两个分钟序列的pearson相关系数
                try:
                    corr, _ = stats.pearsonr(stock_aligned, other_aligned)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    continue
            
            if correlations:
                # 取所有相关系数绝对值的均值作为该股票的"日孤雁出群"
                avg_corr = np.mean(correlations)
                daily_results.append({
                    'trade_date': trade_date,
                    'instrument': stock,
                    'daily_gu_yan_chu_qun': avg_corr
                })
    
    if not daily_results:
        return pd.DataFrame()
    
    return pd.DataFrame(daily_results)


def calculate_daily_factors(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算日频随波逐流和孤雁出群因子。
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
    
    print("计算随波逐流因子（高低额差相关性）...")
    # 按日期计算高低额差
    all_high_low_diff = []
    for trade_date in df_minute['trade_date'].unique():
        day_minute = df_minute[df_minute['trade_date'] == trade_date]
        day_high_low = calculate_daily_high_low_amount(day_minute, df_daily, market_cap_df, trade_date)
        if not day_high_low.empty:
            all_high_low_diff.append(day_high_low)
    
    if not all_high_low_diff:
        return pd.DataFrame(), pd.DataFrame()
    
    df_high_low_diff = pd.concat(all_high_low_diff, ignore_index=True)
    df_sui_bo = calculate_sui_bo_zhu_liu(df_high_low_diff)
    
    print("计算孤雁出群因子（不分化时刻成交额相关性）...")
    df_gu_yan = calculate_gu_yan_chu_qun(df_minute)
    
    return df_sui_bo, df_gu_yan


def calculate_shui_zhong_xing_zhou_factor(df_sui_bo: pd.DataFrame, df_gu_yan: pd.DataFrame) -> pd.DataFrame:
    """
    合成水中行舟因子。
    
    水中行舟 = (随波逐流 + 孤雁出群) / 2
    
    随波逐流：正向因子，20日滚动
    孤雁出群：负向因子，但代码中取绝对值后也是正向处理，(20日均值+20日标准差)/2
    """
    if df_sui_bo.empty and df_gu_yan.empty:
        return pd.DataFrame()
    
    # 处理随波逐流因子（20日滚动）
    if not df_sui_bo.empty:
        df_sui_bo = df_sui_bo.sort_values(['instrument', 'trade_date']).reset_index(drop=True)
        df_sui_bo['sui_bo_zhu_liu_20'] = df_sui_bo.groupby('instrument')['sui_bo_zhu_liu'].transform(
            lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=10).mean()
        )
    
    # 处理孤雁出群因子（(20日均值+20日标准差)/2）
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
        df_merged['shui_zhong_xing_zhou'] = df_merged['sui_bo_zhu_liu_20']
    else:
        df_merged = df_sui_bo.merge(
            df_gu_yan[['trade_date', 'instrument', 'gu_yan_chu_qun_20']],
            on=['trade_date', 'instrument'],
            how='outer'
        )
        df_merged['date'] = pd.to_datetime(df_merged['trade_date'])
        
        # 填充缺失值
        df_merged['sui_bo_zhu_liu_20'] = df_merged['sui_bo_zhu_liu_20'].fillna(0)
        df_merged['gu_yan_chu_qun_20'] = df_merged['gu_yan_chu_qun_20'].fillna(0)
        
        # 合成水中行舟因子 = (随波逐流 + 孤雁出群) / 2
        df_merged['shui_zhong_xing_zhou'] = (
            df_merged['sui_bo_zhu_liu_20'] + df_merged['gu_yan_chu_qun_20']
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
    
    print(f"=== {FACTOR_NAME}（水中行舟）因子计算 - 研报定义复现版 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print()
    print("【研报核心逻辑】")
    print("随波逐流: 股价高位时，'高低额差'与其他股票的spearman相关系数绝对值")
    print("孤雁出群: 市场不分化时，成交额与其他股票的pearson相关系数绝对值")
    print("水中行舟: (随波逐流 + 孤雁出群) / 2")
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
    
    # 步骤1：计算日频因子
    print("\n【步骤1】计算日频随波逐流和孤雁出群因子...")
    start_time_calc = time.time()
    df_sui_bo, df_gu_yan = calculate_daily_factors(effective_start, end_date)
    calc_time = time.time() - start_time_calc
    
    if df_sui_bo.empty and df_gu_yan.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成:")
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

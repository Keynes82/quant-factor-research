# A03 球队硬币因子 - 完整版（三维度合成）
# 来源：方正证券《个股动量效应的识别及"球队硬币"因子构建》（2022-06-11）
# 档案编号：FQ-20260318-006
# 版本：v3.3 - 修正日内波动率定义
# 修复：日内波动率改为20天日内收益率序列的std（而非分钟std的时序std）
#
# 因子构建逻辑：
#   借鉴Moskowitz(2021)的"球队-硬币"理论：
#   - "硬币型"股票（可知性高）：波动率低、换手稳定 → 预期动量 → 收益率×(-1)
#   - "球队型"股票（可知性低）：波动率高、换手变化大 → 预期反转 → 收益率保持不变
#
#   球队硬币因子 = (修正日间反转 + 修正日内反转 + 修正隔夜反转) / 3
#   其中每个修正反转因子 = (波动翻转因子 + 换手翻转因子) / 2
#
# 维度说明：
#   1. 修正日间反转：基于日频数据（日间收益率 = close/close[1]-1）
#   2. 修正日内反转：基于分钟数据（日内收益率 = close/open-1，使用完整交易时段9:30-15:00）
#   3. 修正隔夜反转：基于日频数据（隔夜距离 = |隔夜收益率 - 截面均值|）
#
# 更新日志：
#   v3.3 (2026-04-20): 【修正】日内波动率使用20天日内收益率序列的std，对齐研报定义
#   v3.2 (2026-04-08): 修复时间过滤，使用完整交易时段930-1500
#   v3.1 (2026-04-08): 修复变量名错误last_date→last_computed
#   v3.0 (2026-04-08): 初始完整版，三维度合成

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
FACTOR_NAME = "A03"                # 本因子编号（球队硬币）
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
    
    # 统一列名
    if 'team_coin_factor' in df.columns:
        df.rename(columns={'team_coin_factor': factor_name}, inplace=True)
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


# ========== 数据获取函数 ==========

def fetch_stock_minute_data_batch(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """批量读取分钟级数据（用于修正日内反转）
    
    注意：使用完整交易时段 9:30-15:00（与研报一致，不排除开盘收盘时段）
    """
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


def fetch_stock_daily_data_batch(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """批量读取日频数据（用于修正日间+隔夜反转）"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    # 扩展起始日期以计算滞后指标
    start_dt_extended = (pd.to_datetime(start_date) - pd.DateOffset(days=10)).strftime('%Y-%m-%d')
    
    sql = f"""
    SELECT 
        date,
        open,
        high,
        low,
        close,
        volume,
        amount,
        turn,
        instrument
    FROM cn_stock_bar1d
    WHERE date >= TIMESTAMP '{start_dt_extended} 00:00:00'
      AND date <= TIMESTAMP '{end_date} 23:59:59'
      AND instrument IN ('{instrument_list}')
    ORDER BY instrument, date
    """
    
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['instrument'] = df['instrument'].astype(str)
        
        return df
    except Exception as e:
        print(f"SQL查询失败: {e}")
        return pd.DataFrame()


# ========== 维度1：修正日内反转（分钟数据） ==========

def process_intraday_dimension(inst: str, df_min: pd.DataFrame, df_daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    处理单只股票的修正日内反转因子
    基于分钟数据计算日内收益率（close/open - 1）
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < 10:
            return None
        
        df_min = df_min.sort_values('date').copy()
        daily_results = []
        
        for trade_date, day_df in df_min.groupby('trade_date'):
            if len(day_df) < 5:
                continue
            
            day_df = day_df.sort_values('date').reset_index(drop=True)
            
            # 计算日内收益率 = 收盘/开盘 - 1
            daily_open = day_df.iloc[0]['open']
            daily_close = day_df.iloc[-1]['close']
            
            if daily_open <= 0 or pd.isna(daily_open):
                continue
                
            intraday_return = daily_close / daily_open - 1
            
            # 计算分钟收益率波动率
            day_df['minute_return'] = day_df['close'].pct_change()
            intraday_volatility = day_df['minute_return'].std()
            
            if pd.isna(intraday_volatility):
                continue
            
            daily_results.append({
                'date': pd.to_datetime(trade_date),
                'instrument': inst,
                'intraday_return': intraday_return,
                'intraday_volatility': intraday_volatility
            })
        
        if not daily_results:
            return None
        
        df_result = pd.DataFrame(daily_results)
        
        # 合并日频数据（换手率）
        if df_daily is not None and not df_daily.empty:
            df_daily_inst = df_daily[df_daily['instrument'] == inst].copy()
            if not df_daily_inst.empty:
                df_daily_inst['date'] = pd.to_datetime(df_daily_inst['date']).dt.normalize()
                df_result['date'] = pd.to_datetime(df_result['date']).dt.normalize()
                df_result = df_result.merge(df_daily_inst[['date', 'turn']], on='date', how='left')
                df_result['turn_change'] = df_result['turn'].diff()
            else:
                df_result['turn'] = np.nan
                df_result['turn_change'] = np.nan
        else:
            df_result['turn'] = np.nan
            df_result['turn_change'] = np.nan
        
        return df_result
        
    except Exception as e:
        return None


def calculate_intraday_factor(df_intraday: pd.DataFrame) -> pd.DataFrame:
    """
    计算修正日内反转因子（向量化）
    """
    if df_intraday is None or df_intraday.empty:
        return pd.DataFrame()
    
    df = df_intraday.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    min_periods = min(10, FACTOR_WINDOW)
    
    # 20日滚动均值
    df['intraday_return_mean'] = df.groupby('instrument')['intraday_return'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    # 【修正v3.3】日内波动率定义：使用20天日内收益率序列的std（而非分钟std的时序std）
    df['intraday_vol_std'] = df.groupby('instrument')['intraday_return'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
    )
    df['turn_change_mean'] = df.groupby('instrument')['turn_change'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 截面均值
    date_stats = df.groupby('date').agg({
        'intraday_vol_std': 'mean',
        'turn_change': 'mean'
    }).rename(columns={
        'intraday_vol_std': 'vol_market_mean',
        'turn_change': 'turn_change_market_mean'
    })
    
    df = df.merge(date_stats, left_on='date', right_index=True, how='left')
    
    # 判断硬币型/球队型
    df['is_coin_vol'] = df['intraday_vol_std'] < df['vol_market_mean']
    df['is_coin_turn'] = df['turn_change'] < df['turn_change_market_mean']
    
    # 波动翻转
    df['vol_flip_intraday'] = np.where(df['is_coin_vol'],
                                       -df['intraday_return_mean'],
                                       df['intraday_return_mean'])
    
    # 换手翻转
    df['turn_flip_intraday'] = np.where(df['is_coin_turn'],
                                        -df['intraday_return_mean'],
                                        df['intraday_return_mean'])
    
    # 修正日内反转 = (波动翻转 + 换手翻转) / 2
    df['revised_intraday_reversal'] = (df['vol_flip_intraday'] + df['turn_flip_intraday']) / 2
    
    df_result = df[df['revised_intraday_reversal'].notna()].copy()
    return df_result[['date', 'instrument', 'revised_intraday_reversal']]


# ========== 维度2：修正日间反转（日频数据） ==========

def calculate_daytime_factor(df_daily_all: pd.DataFrame) -> pd.DataFrame:
    """
    计算修正日间反转因子（基于日频数据）
    
    构建步骤：
    1. 计算日间收益率 = close_t / close_{t-1} - 1
    2. 计算日间波动率 = 20日滚动std(日间收益率)
    3. 计算换手率变化量 = turn_t - turn_{t-1}
    4. 截面均值比较，判断硬币型/球队型
    5. 波动翻转 + 换手翻转，等权合成
    """
    if df_daily_all is None or df_daily_all.empty:
        return pd.DataFrame()
    
    df = df_daily_all.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算日间收益率（需要前一日收盘价）
    df['close_lag1'] = df.groupby('instrument')['close'].shift(1)
    df['daytime_return'] = df['close'] / df['close_lag1'] - 1
    
    # 计算换手率变化量
    df['turn_lag1'] = df.groupby('instrument')['turn'].shift(1)
    df['turn_change'] = df['turn'] - df['turn_lag1']
    
    min_periods = min(10, FACTOR_WINDOW)
    
    # 20日滚动均值和波动率
    df['daytime_return_mean'] = df.groupby('instrument')['daytime_return'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    df['daytime_vol_std'] = df.groupby('instrument')['daytime_return'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
    )
    df['turn_change_mean'] = df.groupby('instrument')['turn_change'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 截面均值
    date_stats = df.groupby('date').agg({
        'daytime_vol_std': 'mean',
        'turn_change': 'mean'
    }).rename(columns={
        'daytime_vol_std': 'vol_market_mean',
        'turn_change': 'turn_change_market_mean'
    })
    
    df = df.merge(date_stats, left_on='date', right_index=True, how='left')
    
    # 判断硬币型/球队型
    df['is_coin_vol'] = df['daytime_vol_std'] < df['vol_market_mean']
    df['is_coin_turn'] = df['turn_change'] < df['turn_change_market_mean']
    
    # 波动翻转
    df['vol_flip_daytime'] = np.where(df['is_coin_vol'],
                                      -df['daytime_return_mean'],
                                      df['daytime_return_mean'])
    
    # 换手翻转
    df['turn_flip_daytime'] = np.where(df['is_coin_turn'],
                                       -df['daytime_return_mean'],
                                       df['daytime_return_mean'])
    
    # 修正日间反转 = (波动翻转 + 换手翻转) / 2
    df['revised_daytime_reversal'] = (df['vol_flip_daytime'] + df['turn_flip_daytime']) / 2
    
    df_result = df[df['revised_daytime_reversal'].notna()].copy()
    df_result['date'] = pd.to_datetime(df_result['date']).dt.normalize()
    return df_result[['date', 'instrument', 'revised_daytime_reversal']]


# ========== 维度3：修正隔夜反转（日频数据） ==========

def calculate_overnight_factor(df_daily_all: pd.DataFrame) -> pd.DataFrame:
    """
    计算修正隔夜反转因子（基于日频数据）
    
    构建步骤：
    1. 计算隔夜收益率 = open_t / close_{t-1} - 1
    2. 隔夜距离 = |隔夜收益率 - 截面均值|
    3. 隔夜距离波动率 = 20日滚动std(隔夜距离)
    4. 换手距离 = |turn_change(t-1) - 截面均值|
    5. 截面均值比较，判断硬币型/球队型
    6. 波动翻转 + 换手翻转，等权合成
    """
    if df_daily_all is None or df_daily_all.empty:
        return pd.DataFrame()
    
    df = df_daily_all.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算隔夜收益率
    df['close_lag1'] = df.groupby('instrument')['close'].shift(1)
    df['overnight_return'] = df['open'] / df['close_lag1'] - 1
    
    # 计算换手率变化量（t-1日）
    df['turn_lag1'] = df.groupby('instrument')['turn'].shift(1)
    df['turn_lag2'] = df.groupby('instrument')['turn'].shift(2)
    df['turn_change_t_1'] = df['turn_lag1'] - df['turn_lag2']
    
    min_periods = min(10, FACTOR_WINDOW)
    
    # 隔夜距离 = |隔夜收益率 - 截面均值|
    # 先计算每日截面均值
    daily_overnight_mean = df.groupby('date')['overnight_return'].mean().reset_index()
    daily_overnight_mean.columns = ['date', 'overnight_market_mean']
    df = df.merge(daily_overnight_mean, on='date', how='left')
    df['overnight_distance'] = (df['overnight_return'] - df['overnight_market_mean']).abs()
    
    # 隔夜距离波动率（20日滚动）
    df['overnight_distance_vol'] = df.groupby('instrument')['overnight_distance'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
    )
    df['overnight_distance_mean'] = df.groupby('instrument')['overnight_distance'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 换手距离 = |turn_change(t-1) - 截面均值|
    daily_turnchange_mean = df.groupby('date')['turn_change_t_1'].mean().reset_index()
    daily_turnchange_mean.columns = ['date', 'turnchange_market_mean']
    df = df.merge(daily_turnchange_mean, on='date', how='left')
    df['turn_distance'] = (df['turn_change_t_1'] - df['turnchange_market_mean']).abs()
    
    # 截面均值比较
    date_stats = df.groupby('date').agg({
        'overnight_distance_vol': 'mean',
        'turn_distance': 'mean'
    }).rename(columns={
        'overnight_distance_vol': 'overnight_vol_market_mean',
        'turn_distance': 'turn_distance_market_mean'
    })
    
    df = df.merge(date_stats, left_on='date', right_index=True, how='left')
    
    # 判断硬币型/球队型
    df['is_coin_overnight_vol'] = df['overnight_distance_vol'] < df['overnight_vol_market_mean']
    df['is_coin_turn_distance'] = df['turn_distance'] < df['turn_distance_market_mean']
    
    # 波动翻转（隔夜距离 × -1 或保持不变）
    df['vol_flip_overnight'] = np.where(df['is_coin_overnight_vol'],
                                        -df['overnight_distance_mean'],
                                        df['overnight_distance_mean'])
    
    # 换手翻转
    df['turn_flip_overnight'] = np.where(df['is_coin_turn_distance'],
                                         -df['overnight_distance_mean'],
                                         df['overnight_distance_mean'])
    
    # 修正隔夜反转 = (波动翻转 + 换手翻转) / 2
    df['revised_overnight_reversal'] = (df['vol_flip_overnight'] + df['turn_flip_overnight']) / 2
    
    df_result = df[df['revised_overnight_reversal'].notna()].copy()
    df_result['date'] = pd.to_datetime(df_result['date']).dt.normalize()
    return df_result[['date', 'instrument', 'revised_overnight_reversal']]


# ========== 三维度合成 ==========

def merge_three_dimensions(df_intraday: pd.DataFrame, 
                           df_daytime: pd.DataFrame, 
                           df_overnight: pd.DataFrame) -> pd.DataFrame:
    """
    合并三个维度的因子，等权合成球队硬币因子
    
    球队硬币因子 = (修正日间反转 + 修正日内反转 + 修正隔夜反转) / 3
    """
    print(f"\n【三维度合成】")
    print(f"  修正日内反转: {len(df_intraday)}条")
    print(f"  修正日间反转: {len(df_daytime)}条")
    print(f"  修正隔夜反转: {len(df_overnight)}条")
    
    # 标准化日期格式
    for df in [df_intraday, df_daytime, df_overnight]:
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.normalize()
        if not df.empty and 'instrument' in df.columns:
            df['instrument'] = df['instrument'].astype(str)
    
    # 合并三个维度
    merged = df_intraday.merge(
        df_daytime, on=['date', 'instrument'], how='outer', suffixes=('', '_daytime')
    )
    merged = merged.merge(
        df_overnight, on=['date', 'instrument'], how='outer', suffixes=('', '_overnight')
    )
    
    # 处理列名冲突
    if 'revised_daytime_reversal' not in merged.columns and 'revised_intraday_reversal_daytime' in merged.columns:
        merged.rename(columns={'revised_intraday_reversal_daytime': 'revised_daytime_reversal'}, inplace=True)
    if 'revised_overnight_reversal' not in merged.columns and 'revised_intraday_reversal_overnight' in merged.columns:
        merged.rename(columns={'revised_intraday_reversal_overnight': 'revised_overnight_reversal'}, inplace=True)
    
    print(f"  合并后数据: {len(merged)}条")
    
    # 统计各维度覆盖情况
    has_intraday = merged['revised_intraday_reversal'].notna()
    has_daytime = merged['revised_daytime_reversal'].notna() if 'revised_daytime_reversal' in merged.columns else pd.Series(False, index=merged.index)
    has_overnight = merged['revised_overnight_reversal'].notna() if 'revised_overnight_reversal' in merged.columns else pd.Series(False, index=merged.index)
    
    print(f"  有日内数据: {has_intraday.sum()}条")
    print(f"  有日间数据: {has_daytime.sum()}条")
    print(f"  有隔夜数据: {has_overnight.sum()}条")
    print(f"  三维度齐全: {(has_intraday & has_daytime & has_overnight).sum()}条")
    
    # 等权合成（只计算有数据的维度）
    def weighted_average(row):
        values = []
        if pd.notna(row.get('revised_intraday_reversal')):
            values.append(row['revised_intraday_reversal'])
        if pd.notna(row.get('revised_daytime_reversal')):
            values.append(row['revised_daytime_reversal'])
        if pd.notna(row.get('revised_overnight_reversal')):
            values.append(row['revised_overnight_reversal'])
        
        if len(values) == 0:
            return np.nan
        return np.mean(values)
    
    merged['team_coin_factor'] = merged.apply(weighted_average, axis=1)
    
    # 只保留有因子值的数据
    result = merged[merged['team_coin_factor'].notna()].copy()
    print(f"  最终有效数据: {len(result)}条")
    
    return result[['date', 'instrument', 'team_coin_factor']]


# ========== 并行处理框架 ==========

def parallel_process_batch(stock_minute_dict: Dict[str, pd.DataFrame],
                           stock_daily_dict: Dict[str, pd.DataFrame],
                           max_workers: int = MAX_WORKERS) -> List[pd.DataFrame]:
    """并行处理一批股票（日内维度）"""
    if not stock_minute_dict:
        return []
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_inst = {
            executor.submit(process_intraday_dimension, inst, 
                           stock_minute_dict.get(inst, pd.DataFrame()),
                           stock_daily_dict.get(inst, pd.DataFrame())): inst 
            for inst in stock_minute_dict.keys()
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
    
    print(f"=== {FACTOR_NAME}（球队硬币）因子计算 - 完整版v3.2 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"\n【更新说明】v3.2: 使用完整交易时段9:30-15:00（不排除开盘收盘）")
    print(f"\n【三维度合成】")
    print(f"  1. 修正日间反转：日频数据（日间收益率）")
    print(f"  2. 修正日内反转：分钟数据（日内收益率，完整时段9:30-15:00）")
    print(f"  3. 修正隔夜反转：日频数据（隔夜距离）")
    
    # 增量计算
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            buffer_days = FACTOR_WINDOW + 10
            effective_start = (pd.to_datetime(last_computed) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"\n【增量模式】检测到最后计算日期: {pd.to_datetime(last_computed).date()}")
            print(f"【增量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
        else:
            buffer_days = FACTOR_WINDOW + 10
            effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"\n【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    else:
        buffer_days = FACTOR_WINDOW + 10
        effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
        print(f"\n【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    
    # 获取所有股票
    print("\n【数据准备】获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        print("未能获取股票列表，流程终止。")
        exit(1)
    
    print(f"获取到 {len(instruments_all)} 只股票")
    
    # ========== 维度1：修正日内反转（分钟数据） ==========
    print("\n" + "="*60)
    print("【维度1】修正日内反转因子（分钟数据）")
    print("="*60)
    
    all_intraday_results = []
    total_batches = (len(instruments_all) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for batch_idx, batch in enumerate(chunk_list(instruments_all, CHUNK_SIZE), start=1):
        print(f"\n处理第 {batch_idx}/{total_batches} 批：{len(batch)} 只股票 ...")
        batch_start = datetime.now()
        
        # 批量获取分钟数据
        df_minute_all = fetch_stock_minute_data_batch(batch, effective_start, end_date)
        # 批量获取日频数据（用于合并换手率）
        df_daily_all = fetch_stock_daily_data_batch(batch, effective_start, end_date)
        
        if df_minute_all is None or df_minute_all.empty:
            continue
        
        # 分组为每只股票的数据
        stock_minute_groups = {
            inst: group.reset_index(drop=True)
            for inst, group in df_minute_all.groupby('instrument')
        }
        
        stock_daily_groups = {}
        if df_daily_all is not None and not df_daily_all.empty:
            stock_daily_groups = {
                inst: group.reset_index(drop=True)
                for inst, group in df_daily_all.groupby('instrument')
            }
        
        # 并行或串行处理
        if use_parallel and len(stock_minute_groups) > 1:
            batch_results = parallel_process_batch(stock_minute_groups, stock_daily_groups, MAX_WORKERS)
        else:
            batch_results = []
            for inst, df_min in stock_minute_groups.items():
                df_daily = stock_daily_groups.get(inst, pd.DataFrame())
                result = process_intraday_dimension(inst, df_min, df_daily)
                if result is not None and not result.empty:
                    batch_results.append(result)
        
        if batch_results:
            df_batch = pd.concat(batch_results, axis=0, ignore_index=True)
            all_intraday_results.append(df_batch)
            batch_time = (datetime.now() - batch_start).total_seconds()
            print(f"第 {batch_idx} 批完成：{len(df_batch)} 行，耗时 {batch_time:.2f}秒")
    
    df_intraday_daily = pd.DataFrame()
    if all_intraday_results:
        df_intraday_daily = pd.concat(all_intraday_results, axis=0, ignore_index=True)
        df_intraday_daily['date'] = pd.to_datetime(df_intraday_daily['date'])
        df_intraday_daily = df_intraday_daily.drop_duplicates(subset=['date', 'instrument'], keep='last')
        print(f"\n日内维度日频数据总计: {len(df_intraday_daily)} 条")
    
    # 计算修正日内反转因子
    df_intraday_factor = pd.DataFrame()
    if not df_intraday_daily.empty:
        print("\n计算修正日内反转因子（向量化滚动）...")
        df_intraday_factor = calculate_intraday_factor(df_intraday_daily)
        print(f"修正日内反转因子: {len(df_intraday_factor)} 条")
    
    # ========== 维度2&3：日间+隔夜（日频数据） ==========
    print("\n" + "="*60)
    print("【维度2&3】修正日间反转 + 修正隔夜反转（日频数据）")
    print("="*60)
    
    # 为日频维度准备扩展日期（需要更多历史数据计算滞后指标）
    start_dt_extended = (pd.to_datetime(effective_start) - pd.DateOffset(days=15)).strftime('%Y-%m-%d')
    
    all_daily_data = []
    for batch_idx, batch in enumerate(chunk_list(instruments_all, CHUNK_SIZE), start=1):
        print(f"\n处理第 {batch_idx}/{total_batches} 批日频数据 ...")
        df_daily_batch = fetch_stock_daily_data_batch(batch, start_dt_extended, end_date)
        if df_daily_batch is not None and not df_daily_batch.empty:
            all_daily_data.append(df_daily_batch)
    
    df_daily_all = pd.DataFrame()
    if all_daily_data:
        df_daily_all = pd.concat(all_daily_data, axis=0, ignore_index=True)
        df_daily_all['date'] = pd.to_datetime(df_daily_all['date'])
        df_daily_all = df_daily_all.drop_duplicates(subset=['date', 'instrument'], keep='last')
        df_daily_all = df_daily_all.sort_values(['instrument', 'date'])
        print(f"\n日频数据总计: {len(df_daily_all)} 条")
    
    # 计算修正日间反转因子
    df_daytime_factor = pd.DataFrame()
    if not df_daily_all.empty:
        print("\n计算修正日间反转因子...")
        df_daytime_factor = calculate_daytime_factor(df_daily_all)
        print(f"修正日间反转因子: {len(df_daytime_factor)} 条")
    
    # 计算修正隔夜反转因子
    df_overnight_factor = pd.DataFrame()
    if not df_daily_all.empty:
        print("\n计算修正隔夜反转因子...")
        df_overnight_factor = calculate_overnight_factor(df_daily_all)
        print(f"修正隔夜反转因子: {len(df_overnight_factor)} 条")
    
    # ========== 三维度合成 ==========
    print("\n" + "="*60)
    print("【最终合成】球队硬币因子 = (日间 + 日内 + 隔夜) / 3")
    print("="*60)
    
    df_final = merge_three_dimensions(df_intraday_factor, df_daytime_factor, df_overnight_factor)
    
    if df_final is None or df_final.empty:
        print("未能计算任何因子值，流程终止。")
        exit(1)
    
    # 过滤到指定日期范围
    df_final['date'] = pd.to_datetime(df_final['date'])
    original_count = len(df_final)
    df_final = df_final[
        (df_final['date'] >= pd.to_datetime(start_date)) &
        (df_final['date'] <= pd.to_datetime(end_date))
    ].reset_index(drop=True)
    print(f"\n日期过滤: {original_count} -> {len(df_final)} 条")
    
    if df_final.empty:
        print("过滤后没有数据需要写入。")
        exit(1)
    
    # 写入数据
    df_to_write = prepare_factor_df_for_write(df_final, FACTOR_NAME)
    
    print("\n【数据写入】...")
    start_time_write = datetime.now()
    written_count = safe_write_to_library(df_to_write, overwrite=overwrite)
    write_time = (datetime.now() - start_time_write).total_seconds()
    
    # 总结
    total_time = time.time() - start_time_total
    print("\n" + "="*60)
    print("=== 完成 ===")
    print("="*60)
    print(f"总耗时: {total_time:.2f}秒")
    print(f"写入数据: {written_count} 条")
    print(f"\n预期表现: Rank IC ~ -9.67% (研报基准)")
    print(f"\n【三维度贡献】")
    print(f"  - 修正日内反转: 基于1分钟K线数据（完整时段9:30-15:00）")
    print(f"  - 修正日间反转: 基于日频收盘价数据")
    print(f"  - 修正隔夜反转: 基于日频开盘价数据")
    print(f"\n【版本】v3.2 (2026-04-08) - 时间过滤修复"

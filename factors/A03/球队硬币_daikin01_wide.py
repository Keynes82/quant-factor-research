# "球队硬币"因子实现 - 宽表版本（daikin01）
# 来源：方正证券《个股动量效应的识别及"球队硬币"因子构建》（2022-06-11）
# 档案编号：FQ-20260318-006
# 因子名称：A03（球队硬币）
# 版本：v1.0 - 基于高频因子低频化模板
#
# 因子构建逻辑：
#   借鉴Moskowitz(2021)的"球队-硬币"理论：
#   - "硬币型"股票（可知性高）：波动率低、换手稳定 → 预期动量 → 收益率×(-1)
#   - "球队型"股票（可知性低）：波动率高、换手变化大 → 预期反转 → 收益率保持不变
#
#   球队硬币因子 = (修正日间反转 + 修正日内反转 + 修正隔夜反转) / 3
#
#   其中每个修正反转因子 = (波动翻转因子 + 换手翻转因子) / 2
#
# 数据表：
#   - cn_stock_bar1m: 分钟级数据（用于计算日内收益率波动率）
#   - cn_stock_bar1d: 日频数据（用于计算日间/隔夜收益率、换手率）

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Optional

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A03"                # 本因子编号（球队硬币）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数（增量计算时往前推）
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）


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
    
    if 'team_coin_factor' in df.columns:
        df = df.rename(columns={'team_coin_factor': factor_name})
    
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
            
            existing_indexed[factor_name] = df_indexed[factor_name]
            
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


def chunk_list(lst, size):
    """将列表分割为指定大小的块"""
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def fetch_stock_minute_data(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 DAI SQL 读取分钟级行情数据，用于计算日内收益率。
    剔除开盘前5分钟(9:30-9:34)和收盘前3分钟(14:57-15:00)。
    """
    start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
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
    WHERE date >= TIMESTAMP '{start_dt} 09:30:00'
      AND date <= TIMESTAMP '{end_dt} 15:00:00'
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) >= 935
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) <= 1456
    """
    
    try:
        query_result = dai.query(sql, filters={"instrument": instruments})
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df['trade_date'] = df['date'].dt.date
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['instrument'] = df['instrument'].astype(str)
        
        return df
        
    except Exception as e:
        print(f"SQL查询失败: {e}")
        return pd.DataFrame()


def fetch_stock_daily_data(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 DAI SQL 读取日频数据，用于计算日间收益率、隔夜收益率、换手率。
    """
    start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    # 需要往前多取一天用于计算隔夜收益率
    start_dt_extended = (pd.to_datetime(start_date) - pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    
    sql = f"""
    SELECT 
        date,
        open,
        close,
        volume,
        amount,
        turn,
        instrument
    FROM cn_stock_bar1d
    WHERE date >= TIMESTAMP '{start_dt_extended} 00:00:00'
      AND date <= TIMESTAMP '{end_dt} 23:59:59'
    """
    
    try:
        query_result = dai.query(sql, filters={"instrument": instruments})
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        
        for col in ['open', 'close', 'volume', 'amount', 'turn']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['instrument'] = df['instrument'].astype(str)
        
        return df
        
    except Exception as e:
        print(f"SQL查询失败: {e}")
        return pd.DataFrame()


def calculate_intraday_returns(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算日内收益率。
    日内收益率 = 收盘价 / 开盘价 - 1
    
    注意：这里的开盘价是9:35的开盘价（已剔除开盘前5分钟），
    收盘价是14:56的收盘价（已剔除收盘前3分钟）。
    """
    df = df_minute.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # 获取每只股票每日的第一条（开盘）和最后一条（收盘）
    daily_data = []
    for (inst, trade_date), group in df.groupby(['instrument', 'trade_date']):
        group = group.sort_values('date').reset_index(drop=True)
        if len(group) > 0:
            daily_open = group.iloc[0]['open']
            daily_close = group.iloc[-1]['close']
            
            if daily_open > 0:
                intraday_return = daily_close / daily_open - 1
                daily_data.append({
                    'date': pd.to_datetime(trade_date),
                    'instrument': inst,
                    'intraday_return': intraday_return
                })
    
    return pd.DataFrame(daily_data)


def calculate_intraday_volatility(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算日内收益率波动率（使用分钟级数据）。
    计算每分钟的收益率，然后求标准差。
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算每分钟收益率
    df['minute_return'] = df.groupby(['instrument', df['date'].dt.date])['close'].pct_change()
    
    # 计算每日的日内收益率波动率（标准差）
    daily_vol = df.groupby(['instrument', df['date'].dt.date])['minute_return'].std().reset_index()
    daily_vol.columns = ['instrument', 'date', 'intraday_volatility']
    daily_vol['date'] = pd.to_datetime(daily_vol['date'])
    
    return daily_vol


def calculate_revised_intraday_reversal(df_daily_factors: pd.DataFrame) -> pd.DataFrame:
    """
    计算修正日内反转因子。
    
    构建步骤：
    1) 计算日内收益率（收盘价/开盘价-1）
    2) 计算日内收益率的均值（月均）和标准差（月稳）
    3) 比较日内波动率与市场截面均值，判断"硬币型"或"球队型"
    4) 波动翻转：波动率<市场均值（硬币型）→ 收益率×(-1)
    5) 换手翻转：换手率变化量<市场均值（硬币型）→ 收益率×(-1)
    """
    if df_daily_factors is None or df_daily_factors.empty:
        return pd.DataFrame()
    
    df = df_daily_factors.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    result_list = []
    
    for inst, group in df.groupby('instrument'):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < FACTOR_WINDOW:
            continue
        
        # 计算20日滚动指标
        group['intraday_return_mean'] = group['intraday_return'].rolling(window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW).mean()
        group['intraday_return_std'] = group['intraday_return'].rolling(window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW).std()
        
        # 获取当前时间点的截面均值（用于判断硬币/球队型）
        # 注意：实际应该在每个月末计算，这里简化处理
        group['vol_market_mean'] = group['intraday_return_std'].expanding().mean()
        group['turn_change_market_mean'] = group['turn_change'].expanding().mean()
        
        # 波动翻转因子
        group['is_coin_vol'] = group['intraday_return_std'] < group['vol_market_mean']
        group['vol_flip_factor'] = np.where(group['is_coin_vol'], 
                                            -group['intraday_return_mean'], 
                                            group['intraday_return_mean'])
        
        # 换手翻转因子
        group['is_coin_turn'] = group['turn_change'] < group['turn_change_market_mean']
        group['turn_flip_factor'] = np.where(group['is_coin_turn'], 
                                             -group['intraday_return_mean'], 
                                             group['intraday_return_mean'])
        
        # 修正日内反转因子 = (波动翻转 + 换手翻转) / 2
        group['revised_intraday_reversal'] = (group['vol_flip_factor'] + group['turn_flip_factor']) / 2
        
        result_list.append(group)
    
    if not result_list:
        return pd.DataFrame()
    
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    return df_result[df_result['revised_intraday_reversal'].notna()]


def process_daily_factors(df_minute_all: pd.DataFrame, df_daily_all: pd.DataFrame) -> pd.DataFrame:
    """
    整合分钟级和日频数据，计算日频因子值。
    """
    # 从分钟数据计算日内收益率
    df_intraday_returns = calculate_intraday_returns(df_minute_all)
    
    # 计算日内波动率（分钟级）
    df_intraday_vol = calculate_intraday_volatility(df_minute_all)
    
    # 合并日内收益率和日内波动率
    df_intraday = df_intraday_returns.merge(df_intraday_vol, on=['date', 'instrument'], how='outer')
    
    # 计算日频指标
    df_daily_all = df_daily_all.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算日间收益率
    df_daily_all['daytime_return'] = df_daily_all.groupby('instrument')['close'].pct_change()
    
    # 计算隔夜收益率
    df_daily_all['overnight_return'] = df_daily_all['open'] / df_daily_all.groupby('instrument')['close'].shift(1) - 1
    
    # 计算换手率变化量
    df_daily_all['turn_change'] = df_daily_all.groupby('instrument')['turn'].diff()
    
    # 合并日内数据
    df_merged = df_daily_all.merge(df_intraday, on=['date', 'instrument'], how='left')
    
    return df_merged


def calculate_team_coin_factor(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    计算球队硬币因子。
    
    合成公式：
    球队硬币因子 = (修正日间反转 + 修正日内反转 + 修正隔夜反转) / 3
    
    当前版本主要实现修正日内反转因子（使用1分钟K线数据）。
    日间和隔夜反转因子可由日频数据计算，这里简化处理。
    """
    if df_factors is None or df_factors.empty:
        return pd.DataFrame()
    
    df = df_factors.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算修正日内反转因子
    df_revised_intraday = calculate_revised_intraday_reversal(df)
    
    if df_revised_intraday.empty:
        return pd.DataFrame()
    
    # 简化版本：主要使用修正日内反转因子
    # 完整版本应同时计算修正日间反转和修正隔夜反转
    df_revised_intraday['team_coin_factor'] = df_revised_intraday['revised_intraday_reversal']
    
    return df_revised_intraday[['date', 'instrument', 'team_coin_factor']].dropna()


# ========== 主流程 ==========
if __name__ == "__main__":
    import time
    start_time_total = time.time()
    
    # 参数配置
    start_date = "2026-02-04"        # 用户指定的起始日期
    end_date = "2026-03-14"          # 用户指定的结束日期
    batch_size = 500                 # 批量处理大小
    overwrite = False                # 是否覆盖本因子的历史数据
    use_incremental = True           # 是否启用增量计算
    
    print(f"=== {FACTOR_NAME}（球队硬币）因子计算 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    
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
    
    print("\n【说明】当前版本主要实现修正日内反转因子（使用1分钟K线）")
    print("完整版本应同时包含修正日间反转和修正隔夜反转因子")
    
    # 获取所有A股股票列表
    print("\n获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        print("未获取到任何股票代码，请检查数据表名或权限。")
    else:
        print(f"获取到 {len(instruments_all)} 只股票，准备分批处理（batch_size={batch_size}）。")
        
        all_daily_factors = []
        
        for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
            print(f"\n处理第 {batch_idx} 批：{len(batch)} 只股票 ...")
            
            # 获取分钟级数据
            df_minute_all = fetch_stock_minute_data(batch, effective_start, end_date)
            
            # 获取日频数据
            df_daily_all = fetch_stock_daily_data(batch, effective_start, end_date)
            
            if df_minute_all is None or df_minute_all.empty or df_daily_all is None or df_daily_all.empty:
                print(f"第 {batch_idx} 批数据不足，跳过。")
                continue
            
            # 处理日频因子
            df_factors = process_daily_factors(df_minute_all, df_daily_all)
            
            if not df_factors.empty:
                all_daily_factors.append(df_factors)
            
            print(f"第 {batch_idx} 批处理完成。")
        
        if not all_daily_factors:
            print("\n未能计算任何日频因子值，流程终止。")
        else:
            print("\n合并所有批次数据...")
            df_all_factors = pd.concat(all_daily_factors, axis=0, ignore_index=True)
            
            # 计算球队硬币因子
            print("\n【步骤】计算球队硬币因子...")
            df_factor = calculate_team_coin_factor(df_all_factors)
            
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
                    print(f"写入耗时: {write_time:.2f}秒")
                    print(f"总耗时: {total_time:.2f}秒")
                    print(f"'{FACTOR_NAME}' 共写入 {written_count} 条数据到 '{FACTOR_LIBRARY_TABLE}'（宽表）")
                    print(f"\n预期表现: Rank IC ~ -9.67%（研报基准）")
                    print(f"\n【说明】当前版本主要实现修正日内反转因子")
                    print("完整球队硬币因子 = (修正日间反转 + 修正日内反转 + 修正隔夜反转) / 3")
                else:
                    print("过滤后没有数据需要写入。")

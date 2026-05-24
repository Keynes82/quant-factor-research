# "多空博弈"因子实现 - 宽表版本（daikin01）
# 来源：方正证券《股票日内多空博弈激烈程度度量与"多空博弈"因子构建》（2023-11-29）
# 档案编号：FQ-20260403-014
# 因子名称：A04（多空博弈）
# 版本：v1.0 - 基于高频因子低频化模板
#
# 因子构建逻辑：
#   1. 成交量博弈-收益率因子：
#      - 将每分钟成交量按过去5分钟收益率排序
#      - 正序成交量 - 倒序成交量
#   2. 振幅博弈因子：
#      - 振幅 = (最高价 - 最低价) / 收盘价
#      - 类似成交量博弈的计算方法
#   3. 多空博弈因子 = (成交量博弈 + 振幅博弈) / 2
#
# 数据表：cn_stock_bar1m (分钟级数据)

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Optional

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A04"                # 本因子编号（多空博弈）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数（增量计算时往前推）
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
RET_WINDOW = 5                     # 收益率计算窗口（5分钟）


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
    
    if 'bull_bear_factor' in df.columns:
        df = df.rename(columns={'bull_bear_factor': factor_name})
    
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
    使用 DAI SQL 读取分钟级行情数据。
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


def calculate_minute_returns(df_minute: pd.DataFrame, window: int = RET_WINDOW) -> pd.DataFrame:
    """
    计算过去N分钟的收益率。
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算分钟收益率（相对于前一分钟）
    df['minute_return'] = df.groupby(['instrument', 'trade_date'])['close'].pct_change()
    
    # 计算过去N分钟累计收益率
    df['past_return'] = df.groupby(['instrument', 'trade_date'])['minute_return'].rolling(
        window=window, min_periods=1
    ).sum().reset_index(level=[0, 1], drop=True)
    
    return df


def calculate_volume_game_factor(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算成交量博弈-收益率因子（日频）。
    
    构建步骤：
    1) 将每分钟成交量按照过去5分钟收益率从小到大排序 → 正序成交量之和
    2) 将每分钟成交量按照过去5分钟收益率从大到小排序 → 倒序成交量之和
    3) 日频因子 = 正序成交量 - 倒序成交量
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'trade_date', 'date']).reset_index(drop=True)
    
    daily_factors = []
    
    for (inst, trade_date), group in df.groupby(['instrument', 'trade_date']):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < 10:  # 数据不足
            continue
        
        # 获取过去5分钟收益率和成交量
        past_returns = group['past_return'].values
        volumes = group['volume'].values
        
        # 按收益率从小到大排序的索引
        asc_idx = np.argsort(past_returns)
        # 按收益率从大到小排序的索引
        desc_idx = np.argsort(past_returns)[::-1]
        
        # 正序成交量之和（收益率从小到大）
        vol_asc_sum = np.sum(volumes[asc_idx])
        # 倒序成交量之和（收益率从大到小）
        vol_desc_sum = np.sum(volumes[desc_idx])
        
        # 日频因子 = 正序成交量 - 倒序成交量
        daily_factor = vol_asc_sum - vol_desc_sum
        
        daily_factors.append({
            'date': pd.to_datetime(trade_date),
            'instrument': inst,
            'volume_game_daily': daily_factor
        })
    
    return pd.DataFrame(daily_factors)


def calculate_amplitude_game_factor(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算振幅博弈因子（日频）。
    
    构建步骤：
    1) 计算每分钟振幅 = (最高价 - 最低价) / 收盘价
    2) 将每分钟振幅按照过去5分钟收益率从小到大排序 → 正序振幅之和
    3) 将每分钟振幅按照过去5分钟收益率从大到小排序 → 倒序振幅之和
    4) 日频因子 = 正序振幅 - 倒序振幅
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'trade_date', 'date']).reset_index(drop=True)
    
    # 计算每分钟振幅
    df['amplitude'] = (df['high'] - df['low']) / df['close']
    
    daily_factors = []
    
    for (inst, trade_date), group in df.groupby(['instrument', 'trade_date']):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < 10:  # 数据不足
            continue
        
        # 获取过去5分钟收益率和振幅
        past_returns = group['past_return'].values
        amplitudes = group['amplitude'].values
        
        # 按收益率从小到大排序的索引
        asc_idx = np.argsort(past_returns)
        # 按收益率从大到小排序的索引
        desc_idx = np.argsort(past_returns)[::-1]
        
        # 正序振幅之和（收益率从小到大）
        amp_asc_sum = np.sum(amplitudes[asc_idx])
        # 倒序振幅之和（收益率从大到小）
        amp_desc_sum = np.sum(amplitudes[desc_idx])
        
        # 日频因子 = 正序振幅 - 倒序振幅
        daily_factor = amp_asc_sum - amp_desc_sum
        
        daily_factors.append({
            'date': pd.to_datetime(trade_date),
            'instrument': inst,
            'amplitude_game_daily': daily_factor
        })
    
    return pd.DataFrame(daily_factors)


def mean_distance_normalize(df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """
    均值距离化处理：截面标准化后减去均值并取绝对值。
    """
    df = df.copy()
    
    # 截面标准化（z-score）
    mean_val = df[factor_col].mean()
    std_val = df[factor_col].std()
    
    if std_val > 0:
        df[factor_col + '_zscore'] = (df[factor_col] - mean_val) / std_val
    else:
        df[factor_col + '_zscore'] = 0
    
    # 减去均值并取绝对值
    df[factor_col + '_norm'] = (df[factor_col + '_zscore'] - df[factor_col + '_zscore'].mean()).abs()
    
    return df


def calculate_bull_bear_factor(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算多空博弈因子。
    
    合成公式：
    1) 成交量博弈因子 = (月均成交量博弈 + 月稳成交量博弈) / 2
    2) 振幅博弈因子 = (月均振幅博弈 + 月稳振幅博弈) / 2
    3) 多空博弈因子 = (成交量博弈 + 振幅博弈) / 2
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    result_list = []
    
    for inst, group in df.groupby('instrument'):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < FACTOR_WINDOW:
            continue
        
        # 对日频因子进行均值距离化处理
        group = mean_distance_normalize(group, 'volume_game_daily')
        group = mean_distance_normalize(group, 'amplitude_game_daily')
        
        # 成交量博弈因子：20日滚动均值和标准差
        group['volume_game_mean'] = group['volume_game_daily_norm'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        group['volume_game_std'] = group['volume_game_daily_norm'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        
        # 振幅博弈因子：20日滚动均值和标准差
        group['amplitude_game_mean'] = group['amplitude_game_daily_norm'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        group['amplitude_game_std'] = group['amplitude_game_daily_norm'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        
        # 合成因子
        group['volume_game'] = (group['volume_game_mean'] + group['volume_game_std']) / 2
        group['amplitude_game'] = (group['amplitude_game_mean'] + group['amplitude_game_std']) / 2
        group['bull_bear_factor'] = (group['volume_game'] + group['amplitude_game']) / 2
        
        result_list.append(group)
    
    if not result_list:
        return pd.DataFrame()
    
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    return df_result[df_result['bull_bear_factor'].notna()]


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
    
    print(f"=== {FACTOR_NAME}（多空博弈）因子计算 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"收益率窗口: {RET_WINDOW}分钟")
    
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
            
            if df_minute_all is None or df_minute_all.empty:
                print(f"第 {batch_idx} 批数据不足，跳过。")
                continue
            
            # 计算过去5分钟收益率
            df_minute_all = calculate_minute_returns(df_minute_all, window=RET_WINDOW)
            
            # 计算成交量博弈-收益率因子（日频）
            df_volume_game = calculate_volume_game_factor(df_minute_all)
            
            # 计算振幅博弈因子（日频）
            df_amplitude_game = calculate_amplitude_game_factor(df_minute_all)
            
            # 合并两个因子
            if not df_volume_game.empty and not df_amplitude_game.empty:
                df_merged = df_volume_game.merge(
                    df_amplitude_game, 
                    on=['date', 'instrument'], 
                    how='inner'
                )
                if not df_merged.empty:
                    all_daily_factors.append(df_merged)
                    print(f"第 {batch_idx} 批处理完成：{len(df_merged)} 条日频数据")
            else:
                print(f"第 {batch_idx} 批因子计算结果为空，跳过。")
        
        if not all_daily_factors:
            print("\n未能计算任何日频因子值，流程终止。")
        else:
            print("\n合并所有批次数据...")
            df_all_factors = pd.concat(all_daily_factors, axis=0, ignore_index=True)
            print(f"合并后日频数据: {len(df_all_factors)} 条")
            
            # 计算多空博弈因子
            print("\n【步骤】计算多空博弈因子（低频化）...")
            df_factor = calculate_bull_bear_factor(df_all_factors)
            
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
                
                if not df_factor.empty:
                    print(f"日期范围: {df_factor['date'].min().date()} ~ {df_factor['date'].max().date()}")
                    
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
                    print(f"\n预期表现: Rank IC ~ -9.73%（研报基准）")
                    print(f"多空年化收益 ~ 40.12%")
                else:
                    print("过滤后没有数据需要写入。")

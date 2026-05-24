# "草木皆兵"因子实现 - 宽表版本（daikin01）- v1.1增强版
# 来源：方正证券《显著效应、极端收益扭曲决策权重和"草木皆兵"因子》（2022-12-13）
# 档案编号：FQ-20260403-008
# 因子名称：A07（草木皆兵）
# 版本：v1.1 - 补充注意力衰减 + 换手率替代个人投资者交易比
#
# 因子构建逻辑（完整复现）：
#   1. 惊恐度计算：
#      - 偏离项 = |个股收益率 - 市场收益率|
#      - 基准项 = |个股收益率| + |市场收益率| + 0.1
#      - 惊恐度 = 偏离项 / 基准项
#
#   2. 波动率加剧（基于1分钟数据）：
#      - 日波动率 = 日内每分钟收益率的标准差
#
#   3. 个人投资者交易比（替代方案）：
#      - 研报原文：单笔<4万元交易视为个人投资者交易
#      - 替代方案：使用换手率（turnover）作为代理变量 【v1.1新增】
#      - 逻辑：高换手股票通常个人投资者占比更高
#
#   4. 注意力衰减（衰减后的惊恐度）：【v1.1新增】
#      - 衰减惊恐度 = 当日惊恐度 - (前1日惊恐度 + 前2日惊恐度)/2
#      - 仅保留正值（t日惊恐度 > 前2日均值的日子）
#      - 逻辑：短暂连续异常收益会引起注意力衰减/恐慌适应
#
#   5. 加权决策分（完整版）：
#      - 加权决策分 = 惊恐度 × 波动率 × 换手率 × 衰减惊恐度 × 收益率 【v1.1修正】
#
#   6. 草木皆兵因子 = (20日加权决策分均值 + 20日加权决策分标准差) / 2
#
# 数据表：
#   - cn_stock_bar1m (分钟级数据): 日内波动率计算
#   - cn_stock_bar1d (日频数据): 个股日收益率、换手率
#   - cn_index_bar1d (指数数据): 中证全指市场收益

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Optional

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A07"                # 本因子编号（草木皆兵）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数（增量计算时往前推）
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
MARKET_INDEX = "000985.SH"         # 中证全指作为市场代表

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
    
    if 'panic_factor' in df.columns:
        df = df.rename(columns={'panic_factor': factor_name})
    
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

def fetch_market_daily_data(start_date: str, end_date: str) -> pd.DataFrame:
    """获取中证全指日频数据作为市场代表。"""
    start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    # 注意：使用 cn_index_bar1d 表获取指数数据
    sql = f"""
    SELECT 
        date,
        close,
        instrument
    FROM cn_index_bar1d
    WHERE instrument = '{MARKET_INDEX}'
      AND date >= '{start_dt}'
      AND date <= '{end_dt}'
    ORDER BY date
    """
    
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            # 如果无法获取中证全指，使用等权平均市场收益作为替代
            print("警告：无法获取中证全指数据，将使用个股等权平均作为市场收益代理")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df['market_return'] = df['close'].pct_change()
        return df[['date', 'market_return']]
        
    except Exception as e:
        print(f"市场数据查询失败: {e}")
        return None

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

def fetch_stock_daily_data(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取个股日频数据（含换手率）。
    【v1.1】增加换手率字段作为个人投资者交易比的替代
    """
    start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    sql = f"""
    SELECT 
        date,
        close,
        turnover,  -- 换手率
        instrument
    FROM cn_stock_bar1d
    WHERE date >= '{start_dt}'
      AND date <= '{end_dt}'
    """
    
    try:
        query_result = dai.query(sql, filters={"instrument": instruments})
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df['daily_return'] = df.groupby('instrument')['close'].pct_change()
        
        # 【v1.1】换手率作为个人投资者交易比的代理变量
        # 研报原文：个人投资者交易比 = 个人投资者成交额 / 总成交额
        # 替代方案：使用换手率，高换手通常意味着更多个人投资者参与
        df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"日频数据查询失败: {e}")
        return pd.DataFrame()

def calculate_daily_volatility(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算日波动率（基于1分钟数据）。
    日波动率 = 日内每分钟收益率的标准差
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算每分钟收益率
    df['minute_return'] = df.groupby(['instrument', 'trade_date'])['close'].pct_change()
    
    # 计算日波动率（日内每分钟收益率的标准差）
    daily_volatility = df.groupby(['instrument', 'trade_date']).agg({
        'minute_return': 'std'
    }).reset_index()
    
    daily_volatility.columns = ['instrument', 'trade_date', 'daily_volatility']
    daily_volatility['date'] = pd.to_datetime(daily_volatility['trade_date'])
    
    return daily_volatility

def calculate_panic_degree(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算惊恐度。
    惊恐度 = |个股收益率 - 市场收益率| / (|个股收益率| + |市场收益率| + 0.1)
    """
    df = stock_df.copy()
    
    if market_df is not None and not market_df.empty:
        # 合并市场收益率
        df = df.merge(market_df, on='date', how='left')
    else:
        # 使用个股等权平均作为市场收益代理
        market_proxy = df.groupby('date')['daily_return'].mean().reset_index()
        market_proxy.columns = ['date', 'market_return']
        df = df.merge(market_proxy, on='date', how='left')
    
    # 计算惊恐度
    # 偏离项 = |个股收益率 - 市场收益率|
    df['deviation'] = (df['daily_return'] - df['market_return']).abs()
    
    # 基准项 = |个股收益率| + |市场收益率| + 0.1
    df['benchmark'] = df['daily_return'].abs() + df['market_return'].abs() + 0.1
    
    # 惊恐度 = 偏离项 / 基准项
    df['panic_degree'] = df['deviation'] / df['benchmark']
    
    return df

def calculate_attention_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    【v1.1新增】计算注意力衰减后的惊恐度。
    
    研报原文（第9页）：
    1) 将t日的惊恐度，减去t-1日和t-2日的惊恐度的均值，得到一个差值
    2) 由于该差值需要作为权重信息来使用，因此要保证指标为正数
    3) 将该差值为负的交易日的数据都替换为空值
    4) 仅保留将t日的惊恐度大于t-1日和t-2日的惊恐度均值的交易日
    5) 将其记为衰减后的"惊恐度"
    
    逻辑：短暂连续异常收益会引起投资者注意力的衰减，或恐慌情绪得到了适应和缓解
    """
    df = df.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算前2日惊恐度的均值（滚动窗口）
    df['panic_degree_lag1'] = df.groupby('instrument')['panic_degree'].shift(1)
    df['panic_degree_lag2'] = df.groupby('instrument')['panic_degree'].shift(2)
    df['panic_degree_lag_mean'] = (df['panic_degree_lag1'] + df['panic_degree_lag2']) / 2
    
    # 衰减惊恐度 = 当日惊恐度 - 前2日均值
    df['panic_degree_decay'] = df['panic_degree'] - df['panic_degree_lag_mean']
    
    # 仅保留正值（t日惊恐度 > 前2日均值的日子），负值替换为NaN
    df['panic_degree_decay'] = df['panic_degree_decay'].where(df['panic_degree_decay'] > 0)
    
    # 清理中间列
    df = df.drop(columns=['panic_degree_lag1', 'panic_degree_lag2', 'panic_degree_lag_mean'])
    
    return df

def calculate_panic_factor(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算草木皆兵因子（低频化）- 【v1.1完整版】。
    
    【v1.1修正】合成公式：
    1) 加权决策分 = 惊恐度 × 波动率 × 换手率 × 衰减惊恐度 × 收益率
       - 惊恐度：偏离市场的程度
       - 波动率：波动率加剧效应
       - 换手率：个人投资者交易比代理变量 【新增】
       - 衰减惊恐度：注意力衰减效应 【新增】
       - 收益率：当日收益方向
    2) 草木皆兵-收益因子 = 20日加权决策分均值
    3) 草木皆兵-波动因子 = 20日加权决策分标准差
    4) 草木皆兵因子 = (草木皆兵-收益 + 草木皆兵-波动) / 2
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    
    df = daily_df.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 【v1.1】计算注意力衰减后的惊恐度
    df = calculate_attention_decay(df)
    
    # 【v1.1修正】加权决策分 = 惊恐度 × 波动率 × 换手率 × 衰减惊恐度 × 收益率
    # 对于衰减惊恐度为NaN的日子，加权决策分也会是NaN（符合研报逻辑：仅保留注意力上升的日子）
    df['weighted_score'] = (df['panic_degree'] * 
                            df['daily_volatility'] * 
                            df['turnover'] * 
                            df['panic_degree_decay'] * 
                            df['daily_return'])
    
    result_list = []
    
    for inst, group in df.groupby('instrument'):
        group = group.sort_values('date').reset_index(drop=True)
        
        # 【v1.1】研报要求：只要每月加权决策分数据足够5条，就可以计算
        # 这里放宽到最小5条数据即可计算（原为FACTOR_WINDOW=20）
        min_periods = 5
        
        # 计算20日滚动均值和标准差（草木皆兵-收益 和 草木皆兵-波动）
        group['weighted_mean'] = group['weighted_score'].rolling(
            window=FACTOR_WINDOW, min_periods=min_periods
        ).mean()
        group['weighted_std'] = group['weighted_score'].rolling(
            window=FACTOR_WINDOW, min_periods=min_periods
        ).std()
        
        # 草木皆兵因子 = (均值 + 标准差) / 2 【等权合成】
        group['panic_factor'] = (group['weighted_mean'] + group['weighted_std']) / 2
        
        result_list.append(group)
    
    if not result_list:
        return pd.DataFrame()
    
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    return df_result[df_result['panic_factor'].notna()]

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
    
    print(f"=== {FACTOR_NAME}（草木皆兵）因子计算 - v1.1增强版 ===")
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
    
    print("\n【因子逻辑-v1.1增强版】")
    print("1) 计算惊恐度 = |个股收益率 - 市场收益率| / (|个股收益率| + |市场收益率| + 0.1)")
    print("2) 基于1分钟数据计算日波动率（波动率加剧效应）")
    print("3) 使用换手率作为个人投资者交易比代理变量 【新增】")
    print("4) 计算注意力衰减后的惊恐度（仅保留正值） 【新增】")
    print("5) 加权决策分 = 惊恐度 × 波动率 × 换手率 × 衰减惊恐度 × 收益率 【修正】")
    print("6) 草木皆兵因子 = (20日加权决策分均值 + 20日加权决策分标准差) / 2")
    
    # 获取市场数据
    print("\n获取市场数据（中证全指）...")
    market_df = fetch_market_daily_data(effective_start, end_date)
    if market_df is None:
        print("将使用个股等权平均作为市场收益代理")
    
    # 获取所有A股股票列表
    print("\n获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        print("未获取到任何股票代码，请检查数据表名或权限。")
    else:
        print(f"获取到 {len(instruments_all)} 只股票，准备分批处理（batch_size={batch_size}）。（仅展示前10只）")
        
        all_daily_factors = []
        
        for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
            print(f"\n处理第 {batch_idx} 批：{len(batch)} 只股票 ...")
            
            # 获取分钟级数据
            df_minute_all = fetch_stock_minute_data(batch, effective_start, end_date)
            
            if df_minute_all is None or df_minute_all.empty:
                print(f"第 {batch_idx} 批分钟数据不足，跳过。")
                continue
            
            # 计算日波动率
            df_volatility = calculate_daily_volatility(df_minute_all)
            
            if df_volatility.empty:
                print(f"第 {batch_idx} 批波动率计算为空，跳过。")
                continue
            
            # 获取日频数据（收益率 + 换手率）【v1.1】
            df_daily = fetch_stock_daily_data(batch, effective_start, end_date)
            
            if df_daily is None or df_daily.empty:
                print(f"第 {batch_idx} 批日频数据不足，跳过。")
                continue
            
            # 合并波动率和日频数据
            df_merged = df_volatility.merge(
                df_daily, 
                on=['instrument', 'date'], 
                how='inner'
            )
            
            if df_merged.empty:
                print(f"第 {batch_idx} 批数据合并为空，跳过。")
                continue
            
            # 计算惊恐度
            df_panic = calculate_panic_degree(df_merged, market_df)
            
            if not df_panic.empty:
                all_daily_factors.append(df_panic)
                print(f"第 {batch_idx} 批处理完成：{len(df_panic)} 条日频数据")
            else:
                print(f"第 {batch_idx} 批惊恐度计算为空，跳过。")
        
        if not all_daily_factors:
            print("\n未能计算任何日频因子值，流程终止。")
        else:
            print("\n合并所有批次数据...")
            df_all_factors = pd.concat(all_daily_factors, axis=0, ignore_index=True)
            print(f"合并后日频数据: {len(df_all_factors)} 条")
            
            # 计算草木皆兵因子
            print("\n【步骤】计算草木皆兵因子（低频化）...")
            df_factor = calculate_panic_factor(df_all_factors)
            
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
                    print(f"\n预期表现: Rank IC ~ -8.90%（研报基准）")
                    print(f"多空年化收益 ~ 32.50%")
                    print(f"月度胜率 ~ 85.71%")
                    print(f"\n【v1.1增强】完整复现研报三个改进方向：")
                    print("  - 波动率加剧（已实现）")
                    print("  - 个人投资者交易比（换手率替代）")
                    print("  - 注意力衰减（新增实现）")
                    print("\n【核心逻辑】显著效应：极端偏离市场的收益会扭曲投资者决策权重")
                    print("当市场平静时个股大跌，会引发更强的恐慌情绪，导致过度卖出，未来补涨。")
                else:
                    print("过滤后没有数据需要写入。")

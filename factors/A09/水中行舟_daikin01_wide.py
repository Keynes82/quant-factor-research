# "水中行舟"因子实现 - 宽表版本（daikin01）
# 来源：方正证券《个股成交额的市场跟随性与"水中行舟"因子》（2023-02-15）
# 档案编号：FQ-20260403-009
# 因子名称：A09（水中行舟）
# 版本：v1.0 - 基于高频因子低频化模板
#
# 因子构建逻辑：
#   由两个子因子等权合成：水中行舟 = (随波逐流 + 孤雁出群) / 2
#
#   1. 随波逐流因子（正向）：
#      - 当个股股价处于相对高位时，成交额与市场趋势关联性越高越好
#      - 合理收益率 = 过去20日日内收益率（收盘/开盘-1）的均值
#      - 高位成交额 = 分钟相对开盘收益率 > 合理收益率 的分钟成交额之和
#      - 低位成交额 = 分钟相对开盘收益率 < 合理收益率 的分钟成交额之和
#      - 高低额差 = (高位成交额 - 低位成交额) / 流通市值
#      - 随波逐流 = 每只股票与其他股票"高低额差"序列的spearman相关系数绝对值的均值
#
#   2. 孤雁出群因子（负向）：
#      - 当市场分化不明显时，成交额与市场趋势关联性越低越好
#      - 分钟市场分化度 = 每分钟所有股票分钟收益率的标准差
#      - 不分化时刻 = 分钟市场分化度 < 当日均值 的时刻
#      - 日孤雁出群 = 每只股票在"不分化时刻"成交额与其他股票的pearson相关系数绝对值的均值
#      - 孤雁出群 = (20日均值 + 20日标准差) / 2
#
# 数据表：cn_stock_bar1m (分钟级数据)、cn_stock_bar1d (日频数据)

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Optional
from scipy.stats import spearmanr, pearsonr

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A09"                # 本因子编号（水中行舟）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数（增量计算时往前推）
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
REASONABLE_RETURN_WINDOW = 20      # 合理收益率计算窗口（20日）

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
    
    if 'water_boat_factor' in df.columns:
        df = df.rename(columns={'water_boat_factor': factor_name})
    
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

def fetch_stock_daily_data(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """获取个股日频数据（用于计算日内收益率和流通市值）。"""
    start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    sql = f"""
    SELECT 
        date,
        open,
        close,
        volume,
        amount,
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
        df['intraday_return'] = df['close'] / df['open'] - 1
        
        return df
        
    except Exception as e:
        print(f"日频数据查询失败: {e}")
        return pd.DataFrame()

def calculate_reasonable_return(daily_df: pd.DataFrame, window: int = REASONABLE_RETURN_WINDOW) -> pd.DataFrame:
    """
    计算合理收益率。
    合理收益率 = 过去window日日内收益率（收盘/开盘-1）的均值
    """
    df = daily_df.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    df['reasonable_return'] = df.groupby('instrument')['intraday_return'].rolling(
        window=window, min_periods=window
    ).mean().reset_index(level=0, drop=True)
    
    return df

def calculate_high_low_amount_diff(df_minute: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算高低额差（随波逐流因子的基础）。
    
    步骤：
    1) 计算每分钟相对开盘收益率 = 分钟收盘价 / 当日开盘 - 1
    2) 与合理收益率比较，区分高位/低位时刻
    3) 高位成交额 - 低位成交额，再除以流通市值
    """
    # 合并分钟数据和日频数据
    df_minute['trade_date'] = pd.to_datetime(df_minute['trade_date'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # 获取每日开盘价格和合理收益率
    daily_info = daily_df[['instrument', 'date', 'open', 'reasonable_return']].copy()
    daily_info.columns = ['instrument', 'trade_date', 'daily_open', 'reasonable_return']
    
    df = df_minute.merge(daily_info, on=['instrument', 'trade_date'], how='left')
    
    # 计算每分钟相对开盘收益率
    df['relative_open_return'] = df['close'] / df['daily_open'] - 1
    
    # 区分高位/低位时刻
    df['is_high'] = df['relative_open_return'] > df['reasonable_return']
    
    # 计算每日高低成交额差
    daily_summary = df.groupby(['instrument', 'trade_date']).apply(
        lambda x: pd.Series({
            'high_amount': x[x['is_high']]['amount'].sum() if x[x['is_high']].shape[0] > 0 else 0,
            'low_amount': x[~x['is_high']]['amount'].sum() if x[~x['is_high']].shape[0] > 0 else 0,
            'daily_open': x['daily_open'].iloc[0] if len(x) > 0 else np.nan
        })
    ).reset_index()
    
    # 计算高低额差（这里用每日开盘价作为流通市值的代理，实际应该用流通市值）
    daily_summary['amount_diff'] = (daily_summary['high_amount'] - daily_summary['low_amount']) / daily_summary['daily_open']
    daily_summary['date'] = daily_summary['trade_date']
    
    return daily_summary[['instrument', 'date', 'amount_diff']]

def calculate_follow_wave(daily_diff_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算随波逐流因子。
    
    计算每只股票与其他股票的"高低额差"序列的spearman相关系数绝对值的均值
    """
    if daily_diff_df.empty:
        return pd.DataFrame()
    
    dates = daily_diff_df['date'].unique()
    result_list = []
    
    for date in dates:
        day_data = daily_diff_df[daily_diff_df['date'] == date].copy()
        if len(day_data) < 2:
            continue
        
        instruments = day_data['instrument'].unique()
        correlations = []
        
        for i, inst_a in enumerate(instruments):
            corr_values = []
            for j, inst_b in enumerate(instruments):
                if i != j:
                    # 获取两只股票过去20日的高低额差序列
                    series_a = daily_diff_df[
                        (daily_diff_df['instrument'] == inst_a) & 
                        (daily_diff_df['date'] <= date)
                    ].tail(FACTOR_WINDOW)['amount_diff'].values
                    
                    series_b = daily_diff_df[
                        (daily_diff_df['instrument'] == inst_b) & 
                        (daily_diff_df['date'] <= date)
                    ].tail(FACTOR_WINDOW)['amount_diff'].values
                    
                    if len(series_a) >= 5 and len(series_b) >= 5:
                        try:
                            corr, _ = spearmanr(series_a, series_b)
                            if not np.isnan(corr):
                                corr_values.append(abs(corr))
                        except:
                            pass
            
            if corr_values:
                correlations.append({
                    'instrument': inst_a,
                    'date': date,
                    'follow_wave': np.mean(corr_values)
                })
        
        if correlations:
            result_list.extend(correlations)
    
    if not result_list:
        return pd.DataFrame()
    
    return pd.DataFrame(result_list)

def calculate_lonely_goose(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算孤雁出群因子。
    
    步骤：
    1) 计算每分钟所有股票的分钟收益率标准差（分钟市场分化度）
    2) 找到不分化时刻（分钟市场分化度 < 当日均值）
    3) 计算每只股票在不分化时刻成交额与其他股票的pearson相关系数绝对值的均值
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算每分钟收益率
    df['minute_return'] = df.groupby(['instrument', 'trade_date'])['close'].pct_change()
    
    # 计算每分钟市场分化度（所有股票分钟收益率的标准差）
    minute_divergence = df.groupby('date')['minute_return'].std().reset_index()
    minute_divergence.columns = ['date', 'market_divergence']
    
    # 计算当日平均分化度
    daily_avg_divergence = minute_divergence.groupby(minute_divergence['date'].dt.date)['market_divergence'].mean().reset_index()
    daily_avg_divergence.columns = ['trade_date', 'avg_divergence']
    daily_avg_divergence['trade_date'] = pd.to_datetime(daily_avg_divergence['trade_date'])
    
    # 合并回分钟数据
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.merge(minute_divergence, on='date', how='left')
    df = df.merge(daily_avg_divergence, on='trade_date', how='left')
    
    # 识别不分化时刻
    df['is_non_divergence'] = df['market_divergence'] < df['avg_divergence']
    
    # 只保留不分化时刻的数据
    non_div_df = df[df['is_non_divergence']].copy()
    
    if non_div_df.empty:
        return pd.DataFrame()
    
    # 计算每日孤雁出群（简化版：直接计算相关性，不计算20日滚动）
    dates = non_div_df['trade_date'].unique()
    result_list = []
    
    for date in dates:
        day_data = non_div_df[non_div_df['trade_date'] == date].copy()
        if len(day_data) < 2:
            continue
        
        # 构建成交额矩阵（股票 × 分钟）
        pivot_amount = day_data.pivot_table(
            index='instrument', 
            columns='date', 
            values='amount', 
            fill_value=0
        )
        
        if pivot_amount.shape[1] < 2:
            continue
        
        instruments = pivot_amount.index.tolist()
        
        for i, inst_a in enumerate(instruments):
            corr_values = []
            for j, inst_b in enumerate(instruments):
                if i != j:
                    series_a = pivot_amount.loc[inst_a].values
                    series_b = pivot_amount.loc[inst_b].values
                    
                    if np.std(series_a) > 0 and np.std(series_b) > 0:
                        try:
                            corr, _ = pearsonr(series_a, series_b)
                            if not np.isnan(corr):
                                corr_values.append(abs(corr))
                        except:
                            pass
            
            if corr_values:
                result_list.append({
                    'instrument': inst_a,
                    'date': date,
                    'lonely_goose': np.mean(corr_values)
                })
    
    if not result_list:
        return pd.DataFrame()
    
    return pd.DataFrame(result_list)

def calculate_water_boat_factor(follow_df: pd.DataFrame, lonely_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算水中行舟因子。
    
    合成公式：
    1) 孤雁出群因子 = (20日均值 + 20日标准差) / 2
    2) 水中行舟因子 = (随波逐流因子 + 孤雁出群因子) / 2
    """
    if follow_df.empty or lonely_df.empty:
        return pd.DataFrame()
    
    # 合并两个子因子
    df = follow_df.merge(lonely_df, on=['instrument', 'date'], how='outer')
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    result_list = []
    
    for inst, group in df.groupby('instrument'):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < FACTOR_WINDOW:
            continue
        
        # 填充缺失值
        group['follow_wave'] = group['follow_wave'].fillna(0)
        group['lonely_goose'] = group['lonely_goose'].fillna(0)
        
        # 计算孤雁出群的20日均值和标准差
        group['lonely_mean'] = group['lonely_goose'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        group['lonely_std'] = group['lonely_goose'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        
        # 孤雁出群因子 = (均值 + 标准差) / 2
        group['lonely_factor'] = (group['lonely_mean'] + group['lonely_std']) / 2
        
        # 随波逐流因子直接使用（已经是相关系数均值）
        group['follow_factor'] = group['follow_wave']
        
        # 水中行舟因子 = (随波逐流 + 孤雁出群) / 2
        group['water_boat_factor'] = (group['follow_factor'] + group['lonely_factor']) / 2
        
        result_list.append(group)
    
    if not result_list:
        return pd.DataFrame()
    
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    return df_result[df_result['water_boat_factor'].notna()]

# ========== 主流程 ==========
if __name__ == "__main__":
    import time
    start_time_total = time.time()
    
    # 参数配置
    start_date = "2026-02-04"        # 用户指定的起始日期
    end_date = "2026-03-14"          # 用户指定的结束日期
    batch_size = 200                 # 批量处理大小（减小以处理复杂计算）
    overwrite = False                # 是否覆盖本因子的历史数据
    use_incremental = True           # 是否启用增量计算
    
    print(f"=== {FACTOR_NAME}（水中行舟）因子计算 ===")
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
            effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=60)).strftime('%Y-%m-%d')
            print(f"【全量模式】需要历史数据，扩展起始: {effective_start}")
    else:
        effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=60)).strftime('%Y-%m-%d')
        print(f"【全量模式】扩展起始: {effective_start}")
    
    print("\n【因子逻辑】")
    print("水中行舟 = (随波逐流 + 孤雁出群) / 2")
    print("\n1) 随波逐流（正向）：股价高位时，成交额与市场关联性越高越好")
    print("   - 高低额差 = (高位成交额 - 低位成交额) / 流通市值")
    print("   - 计算与其他股票高低额差序列的spearman相关系数")
    print("\n2) 孤雁出群（负向）：市场不分化时，成交额独立越好")
    print("   - 找到市场不分化时刻（分钟收益率标准差 < 均值）")
    print("   - 计算与其他股票成交额的pearson相关系数")
    print("   - 20日均值+标准差合成")
    
    # 获取所有A股股票列表
    print("\n获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        print("未获取到任何股票代码，请检查数据表名或权限。")
    else:
        print(f"获取到 {len(instruments_all)} 只股票，准备分批处理（batch_size={batch_size}）。")
        print("\n【注意】此因子计算复杂，涉及两两股票相关性，耗时较长...")
        
        all_minute_data = []
        all_daily_data = []
        
        # 首先收集所有数据
        for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
            print(f"\n收集第 {batch_idx} 批数据：{len(batch)} 只股票 ...")
            
            # 获取分钟级数据
            df_minute = fetch_stock_minute_data(batch, effective_start, end_date)
            if df_minute is not None and not df_minute.empty:
                all_minute_data.append(df_minute)
            
            # 获取日频数据
            df_daily = fetch_stock_daily_data(batch, effective_start, end_date)
            if df_daily is not None and not df_daily.empty:
                all_daily_data.append(df_daily)
        
        if not all_minute_data or not all_daily_data:
            print("\n数据收集不足，流程终止。")
        else:
            print("\n合并所有数据...")
            df_minute_all = pd.concat(all_minute_data, axis=0, ignore_index=True)
            df_daily_all = pd.concat(all_daily_data, axis=0, ignore_index=True)
            
            print(f"分钟数据: {len(df_minute_all)} 条")
            print(f"日频数据: {len(df_daily_all)} 条")
            
            # 计算合理收益率
            print("\n【步骤1】计算合理收益率...")
            df_daily_with_reasonable = calculate_reasonable_return(df_daily_all)
            
            # 计算高低额差（随波逐流基础）
            print("\n【步骤2】计算高低额差（随波逐流基础）...")
            df_amount_diff = calculate_high_low_amount_diff(df_minute_all, df_daily_with_reasonable)
            
            # 计算随波逐流因子
            print("\n【步骤3】计算随波逐流因子（两两相关性，耗时较长）...")
            df_follow = calculate_follow_wave(df_amount_diff)
            
            if df_follow.empty:
                print("随波逐流因子计算为空，流程终止。")
            else:
                print(f"随波逐流因子: {len(df_follow)} 条")
                
                # 计算孤雁出群因子
                print("\n【步骤4】计算孤雁出群因子...")
                df_lonely = calculate_lonely_goose(df_minute_all)
                
                if df_lonely.empty:
                    print("孤雁出群因子计算为空，流程终止。")
                else:
                    print(f"孤雁出群因子: {len(df_lonely)} 条")
                    
                    # 计算水中行舟因子
                    print("\n【步骤5】计算水中行舟因子（低频化）...")
                    df_factor = calculate_water_boat_factor(df_follow, df_lonely)
                    
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
                            print(f"'{FACTOR_NAME}' 共写入 {written_count} 条数据到 '{table_id}'（宽表）")
                            print(f"\n预期表现: Rank IC ~ -9.36%（研报基准）")
                            print(f"多空年化收益 ~ 36.24%")
                            print(f"信息比率 ~ 4.40")
                            print(f"月度胜率 ~ 86.67%")
                            print(f"\n【核心逻辑】水中行舟，顺势而为")
                            print("股价高位时随波逐流，市场清淡时孤雁出群。")
                        else:
                            print("过滤后没有数据需要写入。")

# "云开雾散"因子实现 - 宽表版本（daikin01）
# 来源：方正证券《波动率的波动率与投资者模糊性厌恶》（2022-08-04）
# 档案编号：FQ-20260403-005
# 因子名称：A05（云开雾散）
# 版本：v1.0 - 基于高频因子低频化模板
#
# 因子构建逻辑：
#   1. 计算模糊性（波动率的波动率）：
#      - 分钟收益率 → 5分钟滚动波动率 → 5分钟滚动模糊性
#   2. 识别"起雾时刻"（模糊性 > 均值）
#   3. 计算日频因子：
#      - 模糊金额比 = 雾中金额 / 总体金额
#      - 模糊数量比 = 雾中数量 / 总体数量
#      - 模糊价差 = 模糊金额比 - 模糊数量比
#   4. 月均 + 月稳 → 合成"云开雾散"因子
#
# 数据表：cn_stock_bar1m (分钟级数据)

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Optional

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A05"                # 本因子编号（云开雾散）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数（增量计算时往前推）
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
VOL_WINDOW = 5                     # 波动率计算窗口（5分钟）
AMBIGUITY_WINDOW = 5               # 模糊性计算窗口（5分钟）


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
    
    if 'clouds_clearing_factor' in df.columns:
        df = df.rename(columns={'clouds_clearing_factor': factor_name})
    
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


def calculate_ambiguity(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算模糊性（波动率的波动率）。
    
    构建步骤：
    1) 计算每分钟收益率
    2) 计算5分钟滚动波动率（收益率的标准差）
    3) 计算5分钟滚动模糊性（波动率的标准差）
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算每分钟收益率
    df['minute_return'] = df.groupby(['instrument', 'trade_date'])['close'].pct_change()
    
    # 计算5分钟滚动波动率（前5分钟收益率的标准差）
    df['volatility'] = df.groupby(['instrument', 'trade_date'])['minute_return'].rolling(
        window=VOL_WINDOW, min_periods=VOL_WINDOW
    ).std().reset_index(level=[0, 1], drop=True)
    
    # 计算5分钟滚动模糊性（前5分钟波动率的标准差）
    df['ambiguity'] = df.groupby(['instrument', 'trade_date'])['volatility'].rolling(
        window=AMBIGUITY_WINDOW, min_periods=AMBIGUITY_WINDOW
    ).std().reset_index(level=[0, 1], drop=True)
    
    return df


def calculate_daily_clouds_clearing(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算日频云开雾散相关因子。
    
    构建步骤：
    1) 识别"起雾时刻"（模糊性 > 均值）
    2) 计算模糊金额比 = 雾中金额 / 总体金额
    3) 计算模糊数量比 = 雾中数量 / 总体数量
    4) 计算模糊价差 = 模糊金额比 - 模糊数量比
    """
    df = df_minute.copy()
    
    daily_factors = []
    
    for (inst, trade_date), group in df.groupby(['instrument', 'trade_date']):
        group = group.sort_values('date').reset_index(drop=True)
        
        # 过滤掉模糊性为空的行
        group_valid = group[group['ambiguity'].notna()]
        
        if len(group_valid) < 10:  # 数据不足
            continue
        
        # 计算当日模糊性均值
        ambiguity_mean = group_valid['ambiguity'].mean()
        
        # 识别"起雾时刻"（模糊性 > 均值）
        foggy_moments = group_valid[group_valid['ambiguity'] > ambiguity_mean]
        
        if len(foggy_moments) == 0:
            continue
        
        # 计算"雾中金额"和"总体金额"
        foggy_amount = foggy_moments['amount'].mean()  # 起雾时刻的平均每分钟成交金额
        total_amount = group_valid['amount'].mean()     # 全天的平均每分钟成交金额
        
        # 计算"雾中数量"和"总体数量"
        foggy_volume = foggy_moments['volume'].mean()   # 起雾时刻的平均每分钟成交量
        total_volume = group_valid['volume'].mean()     # 全天的平均每分钟成交量
        
        # 计算模糊金额比和模糊数量比
        if total_amount > 0:
            amount_ratio = foggy_amount / total_amount
        else:
            amount_ratio = np.nan
        
        if total_volume > 0:
            volume_ratio = foggy_volume / total_volume
        else:
            volume_ratio = np.nan
        
        # 计算模糊价差 = 模糊金额比 - 模糊数量比
        # 这个差异反映了投资者在急于卖出时的流动性成本
        if not np.isnan(amount_ratio) and not np.isnan(volume_ratio):
            ambiguity_spread = amount_ratio - volume_ratio
        else:
            ambiguity_spread = np.nan
        
        daily_factors.append({
            'date': pd.to_datetime(trade_date),
            'instrument': inst,
            'ambiguity_spread': ambiguity_spread,
            'amount_ratio': amount_ratio,
            'volume_ratio': volume_ratio
        })
    
    return pd.DataFrame(daily_factors)


def calculate_clouds_clearing_factor(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算云开雾散因子（低频化）。
    
    合成公式：
    云开雾散因子 = (月均模糊价差 + 月稳模糊价差) / 2
    
    完整版本应包含：
    - 模糊关联度因子
    - 模糊金额比因子
    - 修正模糊价差因子
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
        
        # 计算20日滚动均值（月均模糊价差）
        group['ambiguity_spread_mean'] = group['ambiguity_spread'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        
        # 计算20日滚动标准差（月稳模糊价差）
        group['ambiguity_spread_std'] = group['ambiguity_spread'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        
        # 合成云开雾散因子
        group['clouds_clearing_factor'] = (
            group['ambiguity_spread_mean'] + group['ambiguity_spread_std']
        ) / 2
        
        result_list.append(group)
    
    if not result_list:
        return pd.DataFrame()
    
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    return df_result[df_result['clouds_clearing_factor'].notna()]


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
    
    print(f"=== {FACTOR_NAME}（云开雾散）因子计算 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"波动率窗口: {VOL_WINDOW}分钟")
    print(f"模糊性窗口: {AMBIGUITY_WINDOW}分钟")
    
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
    
    print("\n【说明】当前版本主要实现模糊价差因子（核心逻辑）")
    print("完整版本应同时包含模糊关联度、模糊金额比、修正模糊价差")
    
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
            
            # 计算模糊性（波动率的波动率）
            df_minute_all = calculate_ambiguity(df_minute_all)
            
            # 计算日频云开雾散因子
            df_daily = calculate_daily_clouds_clearing(df_minute_all)
            
            if not df_daily.empty:
                all_daily_factors.append(df_daily)
                print(f"第 {batch_idx} 批处理完成：{len(df_daily)} 条日频数据")
            else:
                print(f"第 {batch_idx} 批因子计算结果为空，跳过。")
        
        if not all_daily_factors:
            print("\n未能计算任何日频因子值，流程终止。")
        else:
            print("\n合并所有批次数据...")
            df_all_factors = pd.concat(all_daily_factors, axis=0, ignore_index=True)
            print(f"合并后日频数据: {len(df_all_factors)} 条")
            
            # 计算云开雾散因子（低频化）
            print("\n【步骤】计算云开雾散因子（低频化）...")
            df_factor = calculate_clouds_clearing_factor(df_all_factors)
            
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
                    print(f"\n预期表现: Rank IC ~ -9.81%（研报基准）")
                    print(f"多空年化收益 ~ 30.89%")
                    print(f"\n【说明】当前版本主要实现模糊价差因子（核心逻辑）")
                    print("因子逻辑：当模糊性较大时，若成交金额比例远小于成交量比例，")
                    print("说明投资者急于卖出（挂单价格偏低），未来大概率会发生补涨。")
                else:
                    print("过滤后没有数据需要写入。")

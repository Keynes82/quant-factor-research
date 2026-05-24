# "云开雾散"因子实现 - 宽表版本（daikin01）- 完整版v2.1
# 来源：方正证券《波动率的波动率与投资者模糊性厌恶》（2022-08-04）
# 档案编号：FQ-20260403-005
# 因子名称：A05（云开雾散）
# 版本：v2.1 - 完整版（修正模糊价差按研报第11页实现）
#
# 【更新说明】
# v2.0: 完整实现三因子合成框架
# v2.1: 修复修正模糊价差计算逻辑，严格按研报第11页实现：
#       1) 计算个股过去10天日模糊价差的标准差
#       2) 截面（每天所有股票）上负值部分除以标准差进行修正
#       3) 用s1/s2调整保持负值部分数量级一致
#       4) 均修正模糊价差（20日均值）+ 稳模糊价差（原始20日标准差）等权合成
#
# 因子构建逻辑（研报原文）：
#   1. 计算模糊性（波动率的波动率）：
#      - 分钟收益率 → 5分钟滚动波动率 → 5分钟滚动模糊性
#   2. 识别"起雾时刻"（模糊性 > 均值）
#   3. 计算三个日频子因子：
#      - 模糊关联度 = corr(模糊性, 成交金额)
#      - 模糊金额比 = 雾中金额 / 总体金额
#      - 模糊数量比 = 雾中数量 / 总体数量
#      - 原始模糊价差 = 模糊金额比 - 模糊数量比
#      - 修正模糊价差 = 均修正模糊价差 + 稳模糊价差（研报第11页）
#   4. 每个子因子分别计算：月均 + 月稳
#   5. 三因子等权合成 → 云开雾散因子
#
# 研报表现：
#   - 模糊关联度：Rank IC -9.52%
#   - 模糊金额比：Rank IC -6.93%
#   - 修正模糊价差：Rank IC -9.52%
#   - 云开雾散（合成）：Rank IC -9.81%
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
    
    构建步骤（研报第4页）：
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


def calculate_daily_sub_factors(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算日频三个子因子（v2.0完整版）。
    
    研报第3-4页、第6页、第9-12页：
    
    子因子1 - 模糊关联度：
        每天计算模糊性序列与分钟成交金额序列的相关系数
        
    子因子2 - 模糊金额比：
        模糊金额比 = 雾中金额 / 总体金额
        雾中金额 = 起雾时刻的平均每分钟成交金额
        总体金额 = 全天的平均每分钟成交金额
        
    子因子3 - 修正模糊价差：
        模糊价差 = 模糊金额比 - 模糊数量比
        其中模糊数量比 = 雾中数量 / 总体数量
    """
    df = df_minute.copy()
    
    daily_factors = []
    
    for (inst, trade_date), group in df.groupby(['instrument', 'trade_date']):
        group = group.sort_values('date').reset_index(drop=True)
        
        # 过滤掉模糊性为空的行
        group_valid = group[group['ambiguity'].notna()]
        
        if len(group_valid) < 10:  # 数据不足
            continue
        
        # ==================== 子因子1：模糊关联度 ====================
        # 研报第6页：计算模糊性序列与分钟成交金额序列的相关系数
        ambiguity_amount_corr = np.nan
        if group_valid['amount'].std() > 0 and group_valid['ambiguity'].std() > 0:
            ambiguity_amount_corr = group_valid['ambiguity'].corr(group_valid['amount'])
        
        # ==================== 子因子2&3：模糊金额比和模糊数量比 ====================
        # 计算当日模糊性均值（识别"起雾时刻"的阈值）
        ambiguity_mean = group_valid['ambiguity'].mean()
        
        # 识别"起雾时刻"（模糊性 > 均值）
        foggy_moments = group_valid[group_valid['ambiguity'] > ambiguity_mean]
        
        amount_ratio = np.nan
        volume_ratio = np.nan
        ambiguity_spread = np.nan
        
        if len(foggy_moments) > 0:
            # 计算"雾中金额"和"总体金额"
            foggy_amount = foggy_moments['amount'].mean()   # 起雾时刻的平均每分钟成交金额
            total_amount = group_valid['amount'].mean()     # 全天的平均每分钟成交金额
            
            # 计算"雾中数量"和"总体数量"
            foggy_volume = foggy_moments['volume'].mean()   # 起雾时刻的平均每分钟成交量
            total_volume = group_valid['volume'].mean()     # 全天的平均每分钟成交量
            
            # 计算模糊金额比
            if total_amount > 0:
                amount_ratio = foggy_amount / total_amount
            
            # 计算模糊数量比
            if total_volume > 0:
                volume_ratio = foggy_volume / total_volume
            
            # 计算修正模糊价差 = 模糊金额比 - 模糊数量比
            # 研报第9-11页：这个差异反映了投资者在急于卖出时的流动性成本
            if not np.isnan(amount_ratio) and not np.isnan(volume_ratio):
                ambiguity_spread = amount_ratio - volume_ratio
        
        daily_factors.append({
            'date': pd.to_datetime(trade_date),
            'instrument': inst,
            'ambiguity_amount_corr': ambiguity_amount_corr,  # 子因子1：模糊关联度
            'amount_ratio': amount_ratio,                      # 子因子2：模糊金额比
            'volume_ratio': volume_ratio,                      # 中间变量：模糊数量比
            'ambiguity_spread': ambiguity_spread               # 子因子3：修正模糊价差
        })
    
    return pd.DataFrame(daily_factors)


def calculate_clouds_clearing_factor_v2(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算云开雾散因子v2.1（完整版 - 三因子合成，含修正模糊价差完整逻辑）。
    
    研报第11-12页：
    1. 修正模糊价差 = 均修正模糊价差 + 稳模糊价差（等权合成）
       - 均修正模糊价差 = 过去20日修正日模糊价差的均值
       - 稳模糊价差 = 过去20日原始日模糊价差的标准差（注意：不是修正后的标准差）
    2. 三因子等权合成：云开雾散 = (模糊关联度 + 模糊金额比 + 修正模糊价差) / 3
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # ==================== 第一步：计算修正日模糊价差（研报第11页） ====================
    # 1. 计算原始日模糊价差（已在calculate_daily_sub_factors中计算）
    # 2. 计算过去10天的滚动标准差（用于修正）
    # min_periods=5: 允许部分历史数据不足10天的情况，提高数据利用率，同时保证一定的统计稳定性
    df['ambiguity_spread_std_10d'] = df.groupby('instrument')['ambiguity_spread'].rolling(
        window=10, min_periods=5
    ).std().reset_index(level=0, drop=True)
    
    # 3-6. 截面修正逻辑（按日期分组处理）
    def adjust_ambiguity_spread_cross_section(group):
        """
        研报第11页修正逻辑：
        1) 计算截面上日模糊价差为负的部分的和s1
        2) 将负的部分除以过去10天标准差，记为修正日模糊价差
        3) 计算截面上修正日模糊价差为负的部分的和s2
        4) 将修正后的负值部分除以s2，乘以s1，保持数量级一致
        """
        group = group.copy()
        
        # s1: 截面上原始日模糊价差为负的部分的和
        negative_spread = group[group['ambiguity_spread'] < 0]['ambiguity_spread']
        s1 = negative_spread.sum() if len(negative_spread) > 0 else 1e-10  # 避免除零
        
        # 修正日模糊价差：负值部分除以过去10天标准差，正值部分不变
        group['adjusted_spread'] = group['ambiguity_spread']
        
        # 对负值部分进行修正
        negative_mask = group['ambiguity_spread'] < 0
        std_10d = group['ambiguity_spread_std_10d']
        
        # 只有标准差有效且不为零时才修正
        valid_std = negative_mask & std_10d.notna() & (std_10d > 1e-10)
        group.loc[valid_std, 'adjusted_spread'] = group.loc[valid_std, 'ambiguity_spread'] / std_10d[valid_std]
        
        # s2: 截面上修正日模糊价差为负的部分的和
        negative_adjusted = group[group['adjusted_spread'] < 0]['adjusted_spread']
        s2 = negative_adjusted.sum() if len(negative_adjusted) > 0 else 1e-10  # 避免除零
        
        # 保持数量级一致：修正后的负值部分 = 修正后的负值部分 / s2 * s1
        negative_adjusted_mask = group['adjusted_spread'] < 0
        group.loc[negative_adjusted_mask, 'adjusted_spread'] = (
            group.loc[negative_adjusted_mask, 'adjusted_spread'] / s2 * s1
        )
        
        return group
    
    # 按日期分组，进行截面修正
    df = df.groupby('date', group_keys=False).apply(adjust_ambiguity_spread_cross_section)
    
    # ==================== 第二步：计算各子因子（20日滚动） ====================
    result_list = []
    
    for inst, group in df.groupby('instrument'):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < FACTOR_WINDOW:
            continue
        
        # ==================== 子因子1：模糊关联度（月均+月稳） ====================
        group['ambiguity_corr_mean'] = group['ambiguity_amount_corr'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        group['ambiguity_corr_std'] = group['ambiguity_amount_corr'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        group['ambiguity_corr_factor'] = (group['ambiguity_corr_mean'] + group['ambiguity_corr_std']) / 2
        
        # ==================== 子因子2：模糊金额比（月均+月稳） ====================
        group['amount_ratio_mean'] = group['amount_ratio'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        group['amount_ratio_std'] = group['amount_ratio'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        group['amount_ratio_factor'] = (group['amount_ratio_mean'] + group['amount_ratio_std']) / 2
        
        # ==================== 子因子3：修正模糊价差（研报第11页） ====================
        # 均修正模糊价差 = 过去20日修正日模糊价差的均值
        group['adjusted_spread_mean'] = group['adjusted_spread'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        
        # 稳模糊价差 = 过去20日原始日模糊价差的标准差（注意：不是修正后的！）
        group['original_spread_std'] = group['ambiguity_spread'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        
        # 修正模糊价差 = (均修正模糊价差 + 稳模糊价差) / 2
        group['ambiguity_spread_factor'] = (group['adjusted_spread_mean'] + group['original_spread_std']) / 2
        
        # ==================== 三因子等权合成：云开雾散因子 ====================
        group['clouds_clearing_factor'] = (
            group['ambiguity_corr_factor'] + 
            group['amount_ratio_factor'] + 
            group['ambiguity_spread_factor']
        ) / 3
        
        result_list.append(group)
    
    if not result_list:
        return pd.DataFrame()
    
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    
    # 只返回有有效因子值的行
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
    
    print(f"=== {FACTOR_NAME}（云开雾散）因子计算 v2.1完整版 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"波动率窗口: {VOL_WINDOW}分钟")
    print(f"模糊性窗口: {AMBIGUITY_WINDOW}分钟")
    print(f"\n【v2.1更新】修正模糊价差严格按研报第11页实现：")
    print("  1. 模糊关联度因子（模糊性与成交金额相关系数）")
    print("  2. 模糊金额比因子（雾中金额/总体金额）")
    print("  3. 修正模糊价差因子（研报第11页完整修正逻辑）")
    print("     - 个股过去10天标准差调整")
    print("     - 截面负值部分修正（s1/s2数量级调整）")
    print("     - 均修正模糊价差 + 稳模糊价差等权合成")
    print("  → 三因子等权合成云开雾散因子")
    
    # 增量计算
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            effective_start = last_computed
            print(f"\n【增量模式】实际计算起始: {effective_start}")
        else:
            effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=40)).strftime('%Y-%m-%d')
            print(f"\n【全量模式】需要历史数据，扩展起始: {effective_start}")
    else:
        effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=40)).strftime('%Y-%m-%d')
        print(f"\n【全量模式】扩展起始: {effective_start}")
    
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
            
            # 计算日频三个子因子（v2.0完整版）
            df_daily = calculate_daily_sub_factors(df_minute_all)
            
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
            
            # 计算云开雾散因子（v2.1完整版 - 三因子合成）
            print("\n【步骤】计算云开雾散因子（v2.1完整版 - 三因子合成）...")
            df_factor = calculate_clouds_clearing_factor_v2(df_all_factors)
            
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
                    
                    # 显示三个子因子的统计信息
                    print(f"\n【子因子统计】")
                    print(f"  模糊关联度因子: mean={df_factor['ambiguity_corr_factor'].mean():.6f}, std={df_factor['ambiguity_corr_factor'].std():.6f}")
                    print(f"  模糊金额比因子: mean={df_factor['amount_ratio_factor'].mean():.6f}, std={df_factor['amount_ratio_factor'].std():.6f}")
                    print(f"  修正模糊价差因子: mean={df_factor['ambiguity_spread_factor'].mean():.6f}, std={df_factor['ambiguity_spread_factor'].std():.6f}")
                    print(f"  云开雾散因子（合成）: mean={df_factor['clouds_clearing_factor'].mean():.6f}, std={df_factor['clouds_clearing_factor'].std():.6f}")
                    
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
                    print(f"\n预期表现（研报基准）：")
                    print(f"  模糊关联度: Rank IC ~ -9.52%")
                    print(f"  模糊金额比: Rank IC ~ -6.93%")
                    print(f"  修正模糊价差: Rank IC ~ -9.52%")
                    print(f"  云开雾散（合成）: Rank IC ~ -9.81%")
                    print(f"\n因子逻辑：")
                    print("  当模糊性较大时，若成交金额比例远小于成交量比例，")
                    print("  说明投资者急于卖出（挂单价格偏低），未来大概率会发生补涨。")
                else:
                    print("过滤后没有数据需要写入。")
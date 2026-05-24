# "飞蛾扑火"因子实现 - 宽表版本（daikin01）
# 来源：方正证券《个股股价跳跃及其对振幅因子的改进》（2022-09-22）
# 档案编号：FQ-20260403-006
# 因子名称：A06（飞蛾扑火）
# 版本：v1.2 - 补充修正振幅因子2，完整复现研报逻辑（2026-04-11）
#
# 修正说明：
#   v1.0 BUG: 使用expanding.mean()计算历史均值（错误）
#   v1.1 修正: 使用groupby(date).mean()计算每日截面均值（正确）
#   v1.2 补充: 新增修正振幅因子2（基于日频最高/最低价），完整合成飞蛾扑火因子
#
# 因子构建逻辑：
#   1. 月跳跃度因子（基于分钟数据）：
#      - 单利收益率 = 收盘价/前一分钟收盘价 - 1
#      - 连续复利收益率 = ln(收盘价/前一分钟收盘价)
#      - 单复利差 = 单利收益率 - 连续复利收益率
#      - 泰勒残项 = 2 × 单复利差 - 连续复利收益率²
#      - 日跳跃度 = 日内泰勒残项均值
#      - 月跳跃度 = (月均跳跃度 + 月稳跳跃度) / 2
#
#   2. 修正振幅因子1（基于分钟数据-日跳跃度）：
#      - 振幅 = (最高价 - 最低价) / 前一日收盘价
#      - 根据日跳跃度截面均值分类：
#        * 日跳跃度 < 均值：太阳型振幅（振幅 × -1）
#        * 日跳跃度 > 均值：火把型振幅（振幅 × 1）
#      - 修正振幅1 = 20日翻转振幅均值
#
#   3. 修正振幅因子2（基于日频最高/最低价）：【v1.2新增】
#      - 单利收益率 = t日最高价/t-1日最低价 - 1
#      - 连续复利收益率 = ln(t日最高价/t-1日最低价)
#      - 泰勒残项 = 2×单复利差 - 连续复利收益率²
#      - 根据泰勒残项截面均值分类翻转振幅
#      - 修正振幅2 = 20日翻转振幅均值
#
#   4. 飞蛾扑火因子 = (月跳跃度 + (修正振幅1+修正振幅2)/2) / 2 【v1.2修正】
#
# 数据表：cn_stock_bar1m (分钟级数据)

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Optional

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A06"                # 本因子编号（飞蛾扑火）
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
    
    if 'moth_to_flame_factor' in df.columns:
        df = df.rename(columns={'moth_to_flame_factor': factor_name})
    
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

def calculate_daily_jumpiness(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算日跳跃度因子（基于分钟数据）。
    
    构建步骤：
    1) 单利收益率 = 收盘价/前一分钟收盘价 - 1
    2) 连续复利收益率 = ln(收盘价/前一分钟收盘价)
    3) 单复利差 = 单利收益率 - 连续复利收益率
    4) 泰勒残项 = 2 × 单复利差 - 连续复利收益率²
    5) 日跳跃度 = 日内泰勒残项均值
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 计算单利收益率
    df['simple_return'] = df.groupby(['instrument', 'trade_date'])['close'].pct_change()
    
    # 计算连续复利收益率
    df['log_return'] = np.log(df['close'] / df.groupby(['instrument', 'trade_date'])['close'].shift(1))
    
    # 计算单复利差
    df['diff_return'] = df['simple_return'] - df['log_return']
    
    # 计算泰勒残项 = 2 × 单复利差 - 连续复利收益率²
    df['taylor_residual'] = 2 * df['diff_return'] - df['log_return'] ** 2
    
    # 按日和股票聚合，计算日跳跃度
    daily_jumpiness = df.groupby(['instrument', 'trade_date']).agg({
        'taylor_residual': 'mean',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    daily_jumpiness.columns = ['instrument', 'trade_date', 'daily_jumpiness', 'daily_high', 'daily_low', 'daily_close']
    daily_jumpiness['date'] = pd.to_datetime(daily_jumpiness['trade_date'])
    
    return daily_jumpiness

def calculate_daily_amplitude(daily_df: pd.DataFrame, prev_close_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算日振幅。
    振幅 = (最高价 - 最低价) / 前一日收盘价
    """
    df = daily_df.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 获取前一日收盘价
    df['prev_close'] = df.groupby('instrument')['daily_close'].shift(1)
    
    # 计算振幅
    df['amplitude'] = (df['daily_high'] - df['daily_low']) / df['prev_close']
    
    return df

def calculate_corrected_amplitude_2(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    【v1.2新增】修正振幅因子2（基于日频最高/最低价）。
    
    研报原文（第9页）：
    1) 使用"单利"和"连续复利"计算从t-1日最低价到t日最高价的收益率
    2) 泰勒残项 = 2×单复利差 - 连续复利收益率²
    3) 截面均值分类：泰勒残项 < 均值→太阳型(振幅×-1)，泰勒残项 > 均值→火把型(振幅×1)
    4) 修正振幅2 = 20日翻转振幅均值
    
    参数:
        daily_df: 包含日频数据的DataFrame，需有daily_high, daily_low, daily_close列
    返回:
        添加了corrected_amplitude_2列的DataFrame
    """
    df = daily_df.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 获取前一日最低价
    df['prev_low'] = df.groupby('instrument')['daily_low'].shift(1)
    
    # 步骤1：计算从t-1日最低价到t日最高价的收益率
    # 单利收益率 = t日最高价/t-1日最低价 - 1
    df['simple_return_hl'] = df['daily_high'] / df['prev_low'] - 1
    
    # 连续复利收益率 = ln(t日最高价/t-1日最低价)
    df['log_return_hl'] = np.log(df['daily_high'] / df['prev_low'])
    
    # 步骤2：计算泰勒残项 = 2×(单利-复利) - 复利²
    df['diff_return_hl'] = df['simple_return_hl'] - df['log_return_hl']
    df['taylor_residual_hl'] = 2 * df['diff_return_hl'] - df['log_return_hl'] ** 2
    
    # 步骤3：计算每日截面均值（所有股票的泰勒残项均值）
    daily_cross_mean = df.groupby('date')['taylor_residual_hl'].mean().reset_index()
    daily_cross_mean.columns = ['date', 'cross_mean_residual_hl']
    df = df.merge(daily_cross_mean, on='date', how='left')
    
    # 步骤4：根据泰勒残项截面均值分类，翻转振幅
    # 泰勒残项 < 截面均值：太阳型（振幅 × -1）
    # 泰勒残项 > 截面均值：火把型（振幅 × 1）
    df['flipped_amplitude_2'] = np.where(
        df['taylor_residual_hl'] < df['cross_mean_residual_hl'],
        -df['amplitude'],  # 太阳型：翻转
        df['amplitude']    # 火把型：不变
    )
    
    return df

def calculate_moth_to_flame_factor(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算飞蛾扑火因子（低频化）。
    
    【v1.2修正】合成公式：
    1) 月跳跃度 = (月均跳跃度 + 月稳跳跃度) / 2
    2) 修正振幅1 = 基于分钟数据日跳跃度的20日翻转振幅均值
    3) 修正振幅2 = 基于日频最高/最低价的20日翻转振幅均值 【新增】
    4) 修正振幅 = (修正振幅1 + 修正振幅2) / 2 【修正】
    5) 飞蛾扑火因子 = (月跳跃度 + 修正振幅) / 2 【修正】
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    
    df = daily_df.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 【v1.2新增】先计算修正振幅因子2
    df = calculate_corrected_amplitude_2(df)
    
    # 计算每日截面均值（用于修正振幅1）
    daily_cross_mean = df.groupby('date')['daily_jumpiness'].mean().reset_index()
    daily_cross_mean.columns = ['date', 'cross_mean_jumpiness']
    df = df.merge(daily_cross_mean, on='date', how='left')
    
    result_list = []
    
    for inst, group in df.groupby('instrument'):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < FACTOR_WINDOW:
            continue
        
        # 修正振幅1：基于分钟数据日跳跃度的分类翻转
        group['flipped_amplitude'] = np.where(
            group['daily_jumpiness'] < group['cross_mean_jumpiness'],
            -group['amplitude'],  # 太阳型：翻转
            group['amplitude']     # 火把型：不变
        )
        
        # 计算月均跳跃度和月稳跳跃度
        group['monthly_jumpiness_mean'] = group['daily_jumpiness'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        group['monthly_jumpiness_std'] = group['daily_jumpiness'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        
        # 月跳跃度 = (月均 + 月稳) / 2
        group['monthly_jumpiness'] = (group['monthly_jumpiness_mean'] + group['monthly_jumpiness_std']) / 2
        
        # 修正振幅1 = 20日翻转振幅均值
        group['corrected_amplitude_1'] = group['flipped_amplitude'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        
        # 【v1.2新增】修正振幅2 = 20日翻转振幅2均值
        group['corrected_amplitude_2'] = group['flipped_amplitude_2'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        
        # 【v1.2修正】修正振幅 = (修正振幅1 + 修正振幅2) / 2
        group['corrected_amplitude'] = (group['corrected_amplitude_1'] + group['corrected_amplitude_2']) / 2
        
        # 【v1.2修正】飞蛾扑火因子 = (月跳跃度 + 修正振幅) / 2
        group['moth_to_flame_factor'] = (group['monthly_jumpiness'] + group['corrected_amplitude']) / 2
        
        result_list.append(group)
    
    if not result_list:
        return pd.DataFrame()
    
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    return df_result[df_result['moth_to_flame_factor'].notna()]

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
    
    print(f"=== {FACTOR_NAME}（飞蛾扑火）因子计算 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"版本: v1.2（完整复现研报逻辑）")
    
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
    
    print("\n【因子逻辑-v1.2完整版】")
    print("1) 基于分钟数据计算泰勒残项，得到日跳跃度")
    print("2) 修正振幅1：根据日跳跃度截面均值将振幅分类为'太阳型'和'火把型'，翻转后20日均值")
    print("3) 修正振幅2：基于日频最高/最低价计算泰勒残项，截面分类翻转后20日均值 【新增】")
    print("4) 飞蛾扑火因子 = (月跳跃度 + (修正振幅1+修正振幅2)/2) / 2 【修正】")
    
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
            
            # 计算日跳跃度
            df_daily = calculate_daily_jumpiness(df_minute_all)
            
            if df_daily.empty:
                print(f"第 {batch_idx} 批日跳跃度计算为空，跳过。")
                continue
            
            # 计算日振幅
            df_daily = calculate_daily_amplitude(df_daily, None)
            
            if not df_daily.empty:
                all_daily_factors.append(df_daily)
                print(f"第 {batch_idx} 批处理完成：{len(df_daily)} 条日频数据")
            else:
                print(f"第 {batch_idx} 批振幅计算为空，跳过。")
        
        if not all_daily_factors:
            print("\n未能计算任何日频因子值，流程终止。")
        else:
            print("\n合并所有批次数据...")
            df_all_factors = pd.concat(all_daily_factors, axis=0, ignore_index=True)
            print(f"合并后日频数据: {len(df_all_factors)} 条")
            
            # 计算飞蛾扑火因子
            print("\n【步骤】计算飞蛾扑火因子（低频化）...")
            df_factor = calculate_moth_to_flame_factor(df_all_factors)
            
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
                    print(f"多空年化收益 ~ 37.30%")
                    print(f"月度胜率 ~ 87.83%")
                    print(f"\n【v1.2更新】完整复现研报修正振幅1+2合成逻辑")
                    print("【核心逻辑】识别股价跳跃，区分'火把型'（吸引博彩偏好投资者）")
                    print("和'太阳型'（真正基本面向好）振幅，翻转修正后合成因子。")
                else:
                    print("过滤后没有数据需要写入。")

# "适度冒险"因子实现 - 宽表版本（daikin01）
# 来源：方正证券《成交量激增时刻蕴含的alpha信息》（2022-04-12）
# 档案编号：FQ-20260318-009
# 因子名称：A01（适度冒险）
# 版本：v1.0 - 基于高频因子低频化模板
#
# 因子构建逻辑：
#   1. 定义成交量激增时刻：分钟成交量增加量 > mean + std
#   2. 定义"耀眼5分钟"：激增时刻及其随后4分钟
#   3. 计算"耀眼波动率"：耀眼5分钟内收益率的标准差
#   4. 计算"耀眼收益率"：激增时刻的分钟收益率
#   5. 构建"适度"指标：|个股值 - 截面均值|
#   6. 合成"适度冒险"因子：波动率维度 + 收益率维度
#
# 数据表：cn_stock_bar1m
#   字段：date, instrument, open, high, low, close, volume, amount

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A01"                # 本因子编号（适度冒险）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数（增量计算时往前推）
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
DAZZLING_MINUTES = 5               # 耀眼N分钟（研报默认5分钟）


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
    
    if 'moderate_risk_factor' in df.columns:
        df = df.rename(columns={'moderate_risk_factor': factor_name})
    
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


def identify_surge_minutes(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    识别成交量激增时刻。
    
    研报定义：
    1. 计算每分钟成交量相对于上一分钟的增加量
    2. 计算当日成交量增加量的均值(mean)和标准差(std)
    3. 定义成交量增加量 > mean + std 的时刻为"激增时刻"
    """
    df = df_minute.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # 计算成交量增加量（当前分钟 - 上一分钟）
    df['volume_diff'] = df['volume'].diff()
    
    # 计算当日成交量增加量的均值和标准差
    volume_diff_mean = df['volume_diff'].mean()
    volume_diff_std = df['volume_diff'].std()
    
    # 定义激增时刻：volume_diff > mean + std
    threshold = volume_diff_mean + volume_diff_std
    df['is_surge'] = df['volume_diff'] > threshold
    
    return df


def calculate_dazzling_volatility(df_minute: pd.DataFrame, dazzling_minutes: int = DAZZLING_MINUTES) -> float:
    """
    计算日耀眼波动率。
    
    研报定义：
    1. 找到所有"激增时刻"
    2. 对每个激增时刻，取该分钟及其随后(dazzling_minutes-1)分钟
    3. 计算这dazzling_minutes分钟内收益率的标准差（耀眼波动率）
    4. 计算当日所有耀眼波动率的均值
    """
    df = df_minute.copy()
    
    # 计算分钟收益率
    df['return'] = df['close'].pct_change()
    
    # 找到激增时刻的索引
    surge_indices = df[df['is_surge']].index.tolist()
    
    if not surge_indices:
        return np.nan
    
    dazzling_volatilities = []
    
    for idx in surge_indices:
        # 取激增时刻及其随后的(dazzling_minutes-1)分钟
        end_idx = min(idx + dazzling_minutes, len(df))
        window_df = df.loc[idx:end_idx-1]
        
        if len(window_df) >= 2:
            volatility = window_df['return'].std()
            if not np.isnan(volatility):
                dazzling_volatilities.append(volatility)
    
    if not dazzling_volatilities:
        return np.nan
    
    return np.mean(dazzling_volatilities)


def calculate_dazzling_return(df_minute: pd.DataFrame) -> float:
    """
    计算日耀眼收益率。
    
    研报定义：
    1. 找到所有"激增时刻"
    2. 取激增时刻对应的分钟收益率
    3. 计算当日所有耀眼收益率的均值
    """
    df = df_minute.copy()
    
    # 计算分钟收益率
    df['return'] = df['close'].pct_change()
    
    # 找到激增时刻的收益率
    surge_returns = df[df['is_surge']]['return'].dropna()
    
    if surge_returns.empty:
        return np.nan
    
    return surge_returns.mean()


def process_single_stock(inst: str, df_min: pd.DataFrame, trade_date: datetime.date) -> dict:
    """
    处理单只股票的单日数据，计算日频因子值。
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < 5:
            return None
        
        # 识别成交量激增时刻
        df_processed = identify_surge_minutes(df_min)
        
        # 计算日耀眼波动率和日耀眼收益率
        daily_dazzling_vol = calculate_dazzling_volatility(df_processed)
        daily_dazzling_ret = calculate_dazzling_return(df_processed)
        
        return {
            'date': trade_date,
            'instrument': inst,
            'daily_dazzling_volatility': daily_dazzling_vol,
            'daily_dazzling_return': daily_dazzling_ret
        }
        
    except Exception as e:
        print(f"处理股票 {inst} 时出错: {e}")
        return None


def calculate_daily_factor(start_date: str, end_date: str, batch_size: int = 500) -> pd.DataFrame:
    """
    计算日频因子值（日耀眼波动率和日耀眼收益率）。
    """
    print("获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        print("未获取到任何股票代码，请检查数据表名或权限。")
        return pd.DataFrame()
    
    print(f"获取到 {len(instruments_all)} 只股票，准备分批处理（batch_size={batch_size}）。")
    
    all_daily_factors = []
    
    for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
        print(f"处理第 {batch_idx} 批：{len(batch)} 只股票 ...")
        
        df_minute_all = fetch_stock_minute_data(batch, start_date, end_date)
        
        if df_minute_all is None or df_minute_all.empty:
            print(f"第 {batch_idx} 批未返回任何分钟数据，跳过。")
            continue
        
        for (inst, trade_date), group in df_minute_all.groupby(['instrument', 'trade_date']):
            result = process_single_stock(inst, group.reset_index(drop=True), trade_date)
            if result is not None:
                all_daily_factors.append(result)
        
        print(f"第 {batch_idx} 批处理完成。")
    
    if not all_daily_factors:
        return pd.DataFrame()
    
    df_daily = pd.DataFrame(all_daily_factors)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    
    return df_daily


def calculate_moderate_risk_factor(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算适度冒险因子。
    
    研报构建步骤：
    1. 计算适度日耀眼波动率 = |日耀眼波动率 - 截面均值|
    2. 计算适度日耀眼收益率 = |日耀眼收益率 - 截面均值|
    3. 计算20日滚动平均（月均）和标准差（月稳）
    4. 等权合成月耀眼波动率和月耀眼收益率
    5. 等权合成适度冒险因子
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 对每个交易日，计算截面均值
    date_stats = df.groupby('date').agg({
        'daily_dazzling_volatility': 'mean',
        'daily_dazzling_return': 'mean'
    }).rename(columns={
        'daily_dazzling_volatility': 'vol_mean',
        'daily_dazzling_return': 'ret_mean'
    })
    
    df = df.merge(date_stats, left_on='date', right_index=True, how='left')
    
    # 计算适度指标（距离截面均值的绝对值）
    df['moderate_vol'] = (df['daily_dazzling_volatility'] - df['vol_mean']).abs()
    df['moderate_ret'] = (df['daily_dazzling_return'] - df['ret_mean']).abs()
    
    # 对每个股票计算20日滚动平均和标准差
    result_list = []
    
    for inst, group in df.groupby('instrument'):
        group = group.sort_values('date').reset_index(drop=True)
        
        # 20日滚动平均（月均）
        group['monthly_moderate_vol_mean'] = group['moderate_vol'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        group['monthly_moderate_ret_mean'] = group['moderate_ret'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        
        # 20日滚动标准差（月稳）
        group['monthly_moderate_vol_std'] = group['moderate_vol'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        group['monthly_moderate_ret_std'] = group['moderate_ret'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        
        # 等权合成月耀眼波动率和月耀眼收益率
        group['monthly_dazzling_vol'] = (
            group['monthly_moderate_vol_mean'] + group['monthly_moderate_vol_std']
        ) / 2
        group['monthly_dazzling_ret'] = (
            group['monthly_moderate_ret_mean'] + group['monthly_moderate_ret_std']
        ) / 2
        
        # 等权合成适度冒险因子
        group['moderate_risk_factor'] = (
            group['monthly_dazzling_vol'] + group['monthly_dazzling_ret']
        ) / 2
        
        result_list.append(group)
    
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    
    # 只保留有因子值的行
    df_result = df_result[df_result['moderate_risk_factor'].notna()].copy()
    
    return df_result[['date', 'instrument', 'moderate_risk_factor']]


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
    
    print(f"=== {FACTOR_NAME}（适度冒险）因子计算 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"耀眼分钟数: {DAZZLING_MINUTES}分钟")
    
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
    print("\n【步骤1】计算日频因子...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor(effective_start, end_date, batch_size=batch_size)
    calc_time = time.time() - start_time_calc
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成: {len(df_daily)} 条")
        
        # 步骤2：计算适度冒险因子（20日滚动）
        print("\n【步骤2】计算适度冒险因子（20日滚动）...")
        df_factor = calculate_moderate_risk_factor(df_daily)
        
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
                print(f"\n预期表现: Rank IC ~ -8.89%（研报基准）")
            else:
                print("过滤后没有数据需要写入。")

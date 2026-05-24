# "勇攀高峰"因子实现 - 宽表版本（daikin01）
# 来源：方正证券《个股波动率的变动及"勇攀高峰"因子构建》（2022-05-30）
# 档案编号：FQ-20260403-003
# 因子名称：A08（勇攀高峰）
# 版本：v1.0 - 基于高频因子低频化模板
#
# 因子构建逻辑：
#   1. 更优波动率计算（基于最近5分钟OHLC数据）：
#      - 取第t-4、t-3、t-2、t-1、t分钟的OHLC数据（共20个价格）
#      - 更优波动率 = (标准差 / 均值)²
#
#   2. 收益波动比计算：
#      - 收益波动比 = 分钟收益率 / 更优波动率
#
#   3. 识别异常高波动时段：
#      - 当日更优波动率均值 = mean，标准差 = std
#      - 异常高波动 = 更优波动率 >= mean + std
#
#   4. 计算协方差（仅异常高波动时段）：
#      - 协方差 = cov(收益波动比, 更优波动率)
#
#   5. 勇攀高峰因子 = (20日协方差均值 + 20日协方差标准差) / 2
#
# 数据表：cn_stock_bar1m (分钟级数据)

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Tuple, Optional

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A08"                # 本因子编号（勇攀高峰）
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
    
    if 'climb_factor' in df.columns:
        df = df.rename(columns={'climb_factor': factor_name})
    
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

def calculate_better_volatility(df_minute: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    计算"更优波动率"。
    
    计算方法：
    1) 取第t-4、t-3、t-2、t-1、t分钟的OHLC数据（共20个价格）
    2) 计算这20个价格的标准差和均值
    3) 更优波动率 = (标准差 / 均值)²
    
    参数：
        window: 窗口大小（默认5分钟）
    """
    df = df_minute.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # 为每个instrument和trade_date计算更优波动率
    def calc_vol(group):
        group = group.sort_values('date').reset_index(drop=True)
        
        # 展开OHLC数据
        ohlc_data = []
        for _, row in group.iterrows():
            ohlc_data.extend([row['open'], row['high'], row['low'], row['close']])
        
        # 计算滚动更优波动率
        better_vols = []
        for i in range(len(group)):
            if i < window - 1:
                better_vols.append(np.nan)
            else:
                # 取当前及前window-1分钟的OHLC数据
                start_idx = max(0, (i - window + 1) * 4)
                end_idx = (i + 1) * 4
                prices = ohlc_data[start_idx:end_idx]
                
                if len(prices) >= 8:  # 至少2分钟的完整数据
                    mean_price = np.mean(prices)
                    std_price = np.std(prices, ddof=0)
                    if mean_price > 0:
                        better_vol = (std_price / mean_price) ** 2
                        better_vols.append(better_vol)
                    else:
                        better_vols.append(np.nan)
                else:
                    better_vols.append(np.nan)
        
        group['better_volatility'] = better_vols
        
        # 计算分钟收益率
        group['minute_return'] = group['close'].pct_change()
        
        # 计算收益波动比
        group['return_vol_ratio'] = group['minute_return'] / group['better_volatility']
        
        return group
    
    df = df.groupby(['instrument', 'trade_date'], group_keys=False).apply(calc_vol)
    
    return df

def calculate_daily_covariance(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    计算每日协方差（仅异常高波动时段）。
    
    步骤：
    1) 计算当日更优波动率的均值mean和标准差std
    2) 识别异常高波动时段：更优波动率 >= mean + std
    3) 计算这些时段的收益波动比与更优波动率的协方差
    """
    df = df_minute.copy()
    
    # 按日和股票分组计算
    daily_cov_list = []
    
    for (inst, trade_date), group in df.groupby(['instrument', 'trade_date']):
        group = group.dropna(subset=['better_volatility', 'return_vol_ratio'])
        
        if len(group) < 10:  # 需要足够的数据点
            continue
        
        # 计算更优波动率的均值和标准差
        vol_mean = group['better_volatility'].mean()
        vol_std = group['better_volatility'].std()
        
        # 识别异常高波动时段
        high_vol_mask = group['better_volatility'] >= (vol_mean + vol_std)
        high_vol_data = group[high_vol_mask]
        
        if len(high_vol_data) < 3:  # 需要至少3个数据点计算协方差
            continue
        
        # 计算协方差
        cov = np.cov(high_vol_data['return_vol_ratio'], high_vol_data['better_volatility'])[0, 1]
        
        daily_cov_list.append({
            'instrument': inst,
            'trade_date': trade_date,
            'covariance': cov,
            'high_vol_minutes': len(high_vol_data),
            'total_minutes': len(group)
        })
    
    if not daily_cov_list:
        return pd.DataFrame()
    
    df_cov = pd.DataFrame(daily_cov_list)
    df_cov['date'] = pd.to_datetime(df_cov['trade_date'])
    
    return df_cov

def calculate_climb_factor(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算勇攀高峰因子（低频化）。
    
    合成公式：
    勇攀高峰因子 = (20日协方差均值 + 20日协方差标准差) / 2
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    
    df = daily_df.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    result_list = []
    
    for inst, group in df.groupby('instrument'):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < FACTOR_WINDOW:
            continue
        
        # 计算20日滚动均值和标准差
        group['cov_mean'] = group['covariance'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).mean()
        group['cov_std'] = group['covariance'].rolling(
            window=FACTOR_WINDOW, min_periods=FACTOR_WINDOW
        ).std()
        
        # 勇攀高峰因子 = (均值 + 标准差) / 2
        group['climb_factor'] = (group['cov_mean'] + group['cov_std']) / 2
        
        result_list.append(group)
    
    if not result_list:
        return pd.DataFrame()
    
    df_result = pd.concat(result_list, axis=0, ignore_index=True)
    return df_result[df_result['climb_factor'].notna()]

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
    
    print(f"=== {FACTOR_NAME}（勇攀高峰）因子计算 ===")
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
    
    print("\n【因子逻辑】")
    print("1) 更优波动率 = (最近5分钟OHLC数据的标准差/均值)²")
    print("2) 收益波动比 = 分钟收益率 / 更优波动率")
    print("3) 识别异常高波动时段：更优波动率 >= mean + std")
    print("4) 计算这些时段收益波动比与更优波动率的协方差")
    print("5) 勇攀高峰因子 = (20日协方差均值 + 20日协方差标准差) / 2")
    
    # 获取所有A股股票列表
    print("\n获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        print("未获取到任何股票代码，请检查数据表名或权限。")
    else:
        print(f"获取到 {len(instruments_all)} 只股票，准备分批处理（batch_size={batch_size}）。")
        
        all_daily_covs = []
        
        for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
            print(f"\n处理第 {batch_idx} 批：{len(batch)} 只股票 ...")
            
            # 获取分钟级数据
            df_minute_all = fetch_stock_minute_data(batch, effective_start, end_date)
            
            if df_minute_all is None or df_minute_all.empty:
                print(f"第 {batch_idx} 批数据不足，跳过。")
                continue
            
            # 计算更优波动率和收益波动比
            df_better_vol = calculate_better_volatility(df_minute_all)
            
            if df_better_vol.empty:
                print(f"第 {batch_idx} 批更优波动率计算为空，跳过。")
                continue
            
            # 计算每日协方差
            df_daily_cov = calculate_daily_covariance(df_better_vol)
            
            if not df_daily_cov.empty:
                all_daily_covs.append(df_daily_cov)
                print(f"第 {batch_idx} 批处理完成：{len(df_daily_cov)} 条日频数据")
            else:
                print(f"第 {batch_idx} 批协方差计算为空，跳过。")
        
        if not all_daily_covs:
            print("\n未能计算任何日频因子值，流程终止。")
        else:
            print("\n合并所有批次数据...")
            df_all_covs = pd.concat(all_daily_covs, axis=0, ignore_index=True)
            print(f"合并后日频数据: {len(df_all_covs)} 条")
            
            # 计算勇攀高峰因子
            print("\n【步骤】计算勇攀高峰因子（低频化）...")
            df_factor = calculate_climb_factor(df_all_covs)
            
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
                    print(f"\n预期表现: Rank IC ~ 5.62%（研报基准）")
                    print(f"多空年化收益 ~ 19.76%")
                    print(f"月度胜率 ~ 83.02%")
                    print(f"\n【核心逻辑】异常高波动+充足风险补偿 = 勇攀高峰")
                    print("当波动异常高时，能给异常高波动及时提供风险补偿的股票，")
                    print("展现了非凡的能力，这种向好的势头将会长期持续。")
                else:
                    print("过滤后没有数据需要写入。")

# 待著而救因子 - BigQuant宽表版本
# 档案编号: A10
# 因子名称: 待著而救
# 研报来源: 方正证券-多因子选股系列研究之十一
# 研报日期: 2023-06-11
# 目标Rank IC: -9.28%
# 方向: 负向因子（因子值越小，收益越高）
#
# 因子逻辑：大单成交后的跟随效应
# 1. 剔除开盘前15分钟数据
# 2. 找当日成交量最大的10个"海量时刻"
# 3. 筛选间隔>5分钟的"优势时刻"
# 4. 计算优势时刻后5分钟成交量/优势时刻成交量的"跟随系数"
# 5. 日跟随系数 = 日内所有跟随系数均值
# 6. 月均因子=20日均值，月稳因子=20日标准差，等权合成

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A10"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数（增量计算时往前推）
FACTOR_WINDOW = 20                 # 因子滚动窗口（用于低频化平滑）


# ========== 以下函数一般无需修改 ==========

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
    
    if 'factor' in df.columns:
        df = df.rename(columns={'factor': factor_name})
    elif 'smoothed_factor' in df.columns:
        df = df.rename(columns={'smoothed_factor': factor_name})
    
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


def fetch_stock_minute_data(instruments, start_date, end_date):
    """
    使用 DAI SQL 读取分钟级行情数据。
    保留完整交易时段数据（用于计算因子）。
    """
    sql = """
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
    WHERE date >= TIMESTAMP '%s 09:30:00'
      AND date <= TIMESTAMP '%s 15:00:00'
    """ % (start_date, end_date)
    
    try:
        query_result = dai.query(sql, filters={"instrument": instruments})
        df_all = query_result.df()
    except Exception as e:
        print(f"SQL查询失败: {e}")
        return {}

    if df_all is None or df.empty:
        return {}

    df = df_all.copy()
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['time'] = df['date']
        df['date_only'] = df['date'].dt.date
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['instrument'] = df['instrument'].astype(str)
    
    stock_minute_data = {inst: group.reset_index(drop=True) for inst, group in df.groupby('instrument')}
    return stock_minute_data


def find_heavy_volume_moments(df_day, top_n=10):
    """
    找出当日的"海量时刻"（成交量最大的top_n个时刻）
    剔除开盘前15分钟（9:45之前）
    
    Returns:
        list of (timestamp, volume) tuples, sorted by time
    """
    # 只保留9:45之后的数据（剔除开盘前15分钟）
    df_day['time_only'] = df_day['date'].dt.time
    df_day['minute_from_open'] = df_day['date'].dt.hour * 60 + df_day['date'].dt.minute
    # 9:45 = 9*60 + 45 = 585分钟
    df_filtered = df_day[df_day['minute_from_open'] >= 585].copy()
    
    if len(df_filtered) < top_n:
        return []
    
    # 找成交量最大的top_n个时刻
    top_volumes = df_filtered.nlargest(top_n, 'volume')[['date', 'volume']]
    moments = list(zip(top_volumes['date'].values, top_volumes['volume'].values))
    
    # 按时间排序
    moments.sort(key=lambda x: x[0])
    
    return moments


def filter_advantage_moments(heavy_moments, min_interval_minutes=5):
    """
    从"海量时刻"中筛选"优势时刻"
    相邻时刻间隔>min_interval_minutes保留，否则剔除后者
    
    Returns:
        list of (timestamp, volume) tuples
    """
    if not heavy_moments:
        return []
    
    advantage_moments = [heavy_moments[0]]
    
    for i in range(1, len(heavy_moments)):
        current_time = pd.to_datetime(heavy_moments[i][0])
        last_advantage_time = pd.to_datetime(advantage_moments[-1][0])
        
        # 计算时间间隔（分钟）
        interval = (current_time - last_advantage_time).total_seconds() / 60
        
        if interval >= min_interval_minutes:
            advantage_moments.append(heavy_moments[i])
    
    return advantage_moments


def calculate_follow_coefficient(df_day, advantage_moment, follow_window=5):
    """
    计算单个优势时刻的"跟随系数"
    跟随系数 = 优势时刻后follow_window分钟成交量总和 / 优势时刻成交量
    
    Args:
        df_day: 当日分钟数据
        advantage_moment: (timestamp, volume) tuple
        follow_window: 跟随窗口（分钟）
    
    Returns:
        follow_coefficient or None
    """
    adv_time = pd.to_datetime(advantage_moment[0])
    adv_volume = advantage_moment[1]
    
    if adv_volume <= 0:
        return None
    
    # 找优势时刻后follow_window分钟内的数据
    end_time = adv_time + timedelta(minutes=follow_window)
    
    follow_data = df_day[(df_day['date'] > adv_time) & (df_day['date'] <= end_time)]
    
    if len(follow_data) == 0:
        return None
    
    follow_volume = follow_data['volume'].sum()
    
    follow_coeff = follow_volume / adv_volume
    
    return follow_coeff


def process_single_stock(inst, df_min):
    """
    处理单只股票的完整流程，从分钟数据计算日频跟随系数。
    
    Returns:
        DataFrame with columns: ['date', 'instrument', 'daily_follow_coef']
    """
    try:
        if 'date' in df_min.columns:
            df_min['date'] = pd.to_datetime(df_min['date'])
        
        daily_results = []
        
        # 按日期分组处理
        for date, group in df_min.groupby(df_min['date'].dt.date):
            df_day = group.copy()
            
            if len(df_day) < 50:  # 确保数据完整性（至少50分钟数据）
                continue
            
            # 步骤1：找海量时刻
            heavy_moments = find_heavy_volume_moments(df_day, top_n=10)
            
            if not heavy_moments:
                continue
            
            # 步骤2：筛选优势时刻
            advantage_moments = filter_advantage_moments(heavy_moments, min_interval_minutes=5)
            
            if not advantage_moments:
                continue
            
            # 步骤3：计算每个优势时刻的跟随系数
            follow_coeffs = []
            for adv_moment in advantage_moments:
                coeff = calculate_follow_coefficient(df_day, adv_moment, follow_window=5)
                if coeff is not None and not np.isnan(coeff) and not np.isinf(coeff):
                    follow_coeffs.append(coeff)
            
            if not follow_coeffs:
                continue
            
            # 步骤4：日跟随系数 = 日内所有跟随系数均值
            daily_follow_coef = np.mean(follow_coeffs)
            
            if not np.isnan(daily_follow_coef) and not np.isinf(daily_follow_coef):
                daily_results.append({
                    'date': date,
                    'daily_follow_coef': daily_follow_coef,
                    'num_advantage_moments': len(advantage_moments),
                    'num_heavy_moments': len(heavy_moments)
                })
        
        if not daily_results:
            return None
        
        df_daily = pd.DataFrame(daily_results)
        df_daily = df_daily.sort_values('date').reset_index(drop=True)
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_daily['instrument'] = inst
        
        return df_daily[['date', 'instrument', 'daily_follow_coef']]
        
    except Exception as e:
        print(f"Error processing {inst}: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_daily_factor(start_date: str, end_date: str, batch_size: int = 500) -> pd.DataFrame:
    """
    计算日频跟随系数。
    
    Returns:
        DataFrame with columns: ['date', 'instrument', 'daily_follow_coef']
    """
    print("获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        print("未获取到任何股票代码，请检查数据表名或权限。")
        return pd.DataFrame()
    
    print(f"获取到 {len(instruments_all)} 只股票，准备分批处理（batch_size={batch_size}）。")
    
    all_factors = []
    
    for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
        print(f"处理第 {batch_idx} 批：{len(batch)} 只股票 ...")
        try:
            stock_minute_data = fetch_stock_minute_data(batch, start_date, end_date)
        except Exception as e:
            print(f"第 {batch_idx} 批数据拉取失败: {e}")
            continue

        if not stock_minute_data:
            print(f"第 {batch_idx} 批未返回任何分钟数据，跳过。")
            continue

        batch_factors = []
        for inst, df_min in stock_minute_data.items():
            result_df = process_single_stock(inst, df_min)
            if result_df is not None and not result_df.empty:
                batch_factors.append(result_df)

        if len(batch_factors) == 0:
            print(f"第 {batch_idx} 批未计算出任何因子值，跳过写入。")
            continue

        df_batch_all = pd.concat(batch_factors, axis=0, ignore_index=True)
        all_factors.append(df_batch_all)
        print(f"第 {batch_idx} 批处理完成，累计因子数据：{len(df_batch_all)} 行。")

    if len(all_factors) == 0:
        return pd.DataFrame()
    
    df_all = pd.concat(all_factors, axis=0, ignore_index=True)
    df_all = df_all.sort_values(['date', 'instrument']).drop_duplicates(
        subset=['date', 'instrument'], keep='last'
    ).reset_index(drop=True)
    
    return df_all


def calculate_monthly_factor(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    将日频跟随系数低频化为月频因子。
    月均待著而救 = 过去20日跟随系数均值
    月稳待著而救 = 过去20日跟随系数标准差
    待著而救因子 = 月均 + 月稳（等权合成）
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    
    daily_df = daily_df.sort_values(['instrument', 'date']).reset_index(drop=True)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    results = []
    
    for inst, group in daily_df.groupby('instrument'):
        group = group.sort_values('date').reset_index(drop=True)
        
        # 计算20日滚动均值（月均待著而救）
        group['monthly_mean'] = group['daily_follow_coef'].rolling(window=FACTOR_WINDOW, min_periods=10).mean()
        
        # 计算20日滚动标准差（月稳待著而救）
        group['monthly_std'] = group['daily_follow_coef'].rolling(window=FACTOR_WINDOW, min_periods=10).std()
        
        # 合成待著而救因子（等权）
        # 注意：标准差需要处理NaN
        group['factor'] = group['monthly_mean'] + group['monthly_std'].fillna(0)
        
        # 标记月末日期（每个月最后一个交易日）
        group['year_month'] = group['date'].dt.to_period('M')
        group['is_month_end'] = group['year_month'] != group['year_month'].shift(-1)
        
        # 只保留月末数据（低频化）
        monthly_data = group[group['is_month_end']].copy()
        
        if not monthly_data.empty:
            results.append(monthly_data[['date', 'instrument', 'factor']])
    
    if not results:
        return pd.DataFrame()
    
    df_result = pd.concat(results, axis=0, ignore_index=True)
    df_result = df_result.dropna(subset=['factor']).reset_index(drop=True)
    
    return df_result


# ========== 主流程 ==========
if __name__ == "__main__":
    import time
    start_time_total = time.time()
    
    # 参数配置
    start_date = "2026-02-04"
    end_date = "2026-03-14"
    batch_size = 500
    overwrite = False
    use_incremental = True
    
    print(f"=== {FACTOR_NAME} 待著而救因子计算 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"因子方向: 负向（因子值越小，未来收益越高）")
    
    # 增量计算
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            effective_start = last_computed
            print(f"【增量模式】实际计算起始: {effective_start}")
        else:
            effective_start = start_date
            print(f"【全量模式】计算起始: {effective_start}")
    else:
        effective_start = start_date
        print(f"【全量模式】计算起始: {effective_start}")
    
    # 计算日频跟随系数
    print("\n步骤1: 计算日频跟随系数...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor(effective_start, end_date, batch_size=batch_size)
    calc_time = time.time() - start_time_calc
    
    if df_daily is not None and not df_daily.empty:
        print(f"\n日频数据计算完成: {len(df_daily)} 条")
        print(f"日期范围: {df_daily['date'].min().date()} ~ {df_daily['date'].max().date()}")
        print(f"股票数量: {df_daily['instrument'].nunique()}")
        
        # 计算月频因子（低频化）
        print("\n步骤2: 低频化为月频因子...")
        df_monthly = calculate_monthly_factor(df_daily)
        
        if df_monthly is not None and not df_monthly.empty:
            print(f"月频因子计算完成: {len(df_monthly)} 条")
            
            # 过滤到用户指定日期范围
            df_monthly['date'] = pd.to_datetime(df_monthly['date'])
            df_monthly = df_monthly[
                (df_monthly['date'] >= pd.to_datetime(start_date)) & 
                (df_monthly['date'] <= pd.to_datetime(end_date))
            ].reset_index(drop=True)
            
            print(f"过滤后: {len(df_monthly)} 条")
            
            if not df_monthly.empty:
                # 准备宽表格式数据
                df_to_write = prepare_factor_df_for_write(df_monthly, FACTOR_NAME)
                
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
                print(f"'{FACTOR_NAME}' 待著而救因子共写入 {written_count} 条数据到 '{FACTOR_LIBRARY_TABLE}'")
                print(f"\n预期表现:")
                print(f"  - Rank IC: -9.28%")
                print(f"  - 信息比率: 3.51")
                print(f"  - 多空年化收益: 33.16%")
                print(f"  - 月度胜率: 83.87%")
            else:
                print("过滤后没有数据需要写入。")
        else:
            print("月频因子计算失败，无数据写入。")
    else:
        print("未能计算任何日频跟随系数，流程终止。")

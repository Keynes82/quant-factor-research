# 流动性弹性因子 - BigQuant宽表版本
# 档案编号: A15
# 因子名称: 流动性弹性
# 研报来源: 方正证券-多因子选股系列研究之二十三
# 研报日期: 2025-08-27
# 目标Rank IC: -7.15%（预估）
# 方向: 负向因子（因子值越小，收益越高）
#
# 因子逻辑：个股日内成交量分布特征与流动性弹性刻画
# 1. 获取日内分钟级成交量数据
# 2. 计算成交量分布的集中度（基尼系数或赫芬达尔指数）
# 3. 成交量分布越集中，说明流动性弹性越大（少数时刻集中交易）
# 4. 日因子 = 成交量集中度指标
# 5. 月因子 = 过去20日平均值

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A15"                # 本因子编号
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
    保留完整交易时段数据。
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

    if df_all is None or df_all.empty:
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


def calculate_gini_coefficient(values):
    """
    计算基尼系数
    
    基尼系数衡量数据分布的不均衡程度
    - 0：完全均衡（所有值相等）
    - 1：完全不均衡（所有值集中在一个点）
    
    Args:
        values: 数值序列（如成交量序列）
    
    Returns:
        基尼系数
    """
    values = np.array(values)
    values = values[~np.isnan(values)]  # 去除NaN
    
    if len(values) == 0 or np.sum(values) == 0:
        return 0
    
    # 排序
    values_sorted = np.sort(values)
    n = len(values_sorted)
    
    # 计算累积占比
    cumsum = np.cumsum(values_sorted)
    cumsum_normalized = cumsum / cumsum[-1]
    
    # 基尼系数 = 1 - 2 * 曲线下方面积
    # 使用梯形法则计算面积
    x = np.arange(1, n + 1) / n
    area = np.trapz(cumsum_normalized, x)
    gini = 1 - 2 * area
    
    return gini


def calculate_herfindahl_index(values):
    """
    计算赫芬达尔指数（HHI）
    
    HHI = Σ(si^2)，其中si是第i个样本的占比
    - 值越大表示集中度越高
    - 范围：1/n（完全分散）到1（完全集中）
    
    Args:
        values: 数值序列
    
    Returns:
        赫芬达尔指数
    """
    values = np.array(values)
    values = values[~np.isnan(values)]
    
    if len(values) == 0 or np.sum(values) == 0:
        return 0
    
    # 计算占比
    shares = values / np.sum(values)
    
    # 赫芬达尔指数
    hhi = np.sum(shares ** 2)
    
    return hhi


def calculate_volume_concentration_features(volume_series):
    """
    计算成交量集中度特征
    
    Returns:
        dict: 包含基尼系数、赫芬达尔指数等特征
    """
    if len(volume_series) < 10:
        return None
    
    # 去除零值和NaN
    volume_clean = volume_series[volume_series > 0].dropna()
    
    if len(volume_clean) < 10:
        return None
    
    # 基尼系数
    gini = calculate_gini_coefficient(volume_clean.values)
    
    # 赫芬达尔指数
    hhi = calculate_herfindahl_index(volume_clean.values)
    
    # 标准化赫芬达尔指数（消除样本数量影响）
    n = len(volume_clean)
    hhi_normalized = (hhi - 1/n) / (1 - 1/n) if n > 1 else 0
    
    # 最大成交量占比
    max_share = np.max(volume_clean) / np.sum(volume_clean)
    
    # 前10%时间段成交量占比
    top_10_pct_count = max(1, int(n * 0.1))
    top_volumes = np.sort(volume_clean)[-top_10_pct_count:]
    top_10_pct_share = np.sum(top_volumes) / np.sum(volume_clean)
    
    return {
        'gini': gini,
        'hhi': hhi,
        'hhi_normalized': hhi_normalized,
        'max_share': max_share,
        'top_10_pct_share': top_10_pct_share
    }


def process_single_stock(inst, df_min):
    """
    处理单只股票的完整流程，计算日频流动性弹性因子。
    
    日因子 = 成交量集中度（基尼系数或赫芬达尔指数）
    
    Returns:
        DataFrame with columns: ['date', 'instrument', 'factor']
    """
    try:
        if 'date' in df_min.columns:
            df_min['date'] = pd.to_datetime(df_min['date'])
        
        daily_results = []
        
        # 按日期分组处理
        for date, group in df_min.groupby(df_min['date'].dt.date):
            df_day = group.copy().reset_index(drop=True)
            
            if len(df_day) < 20:  # 确保数据完整性
                continue
            
            # 步骤1：计算成交量集中度特征
            concentration_features = calculate_volume_concentration_features(df_day['volume'])
            
            if concentration_features is None:
                continue
            
            # 步骤2：构建日频因子
            # 流动性弹性因子 = 基尼系数（或赫芬达尔指数）
            # 成交量分布越集中，流动性弹性越大
            factor_value = concentration_features['gini']
            
            if not np.isnan(factor_value) and not np.isinf(factor_value):
                daily_results.append({
                    'date': date,
                    'daily_factor': factor_value,
                    'gini': concentration_features['gini'],
                    'hhi': concentration_features['hhi'],
                    'hhi_normalized': concentration_features['hhi_normalized'],
                    'max_share': concentration_features['max_share'],
                    'top_10_pct_share': concentration_features['top_10_pct_share'],
                    'data_points': len(df_day)
                })
        
        if not daily_results:
            return None
        
        df_daily = pd.DataFrame(daily_results)
        df_daily = df_daily.sort_values('date').reset_index(drop=True)
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_daily['instrument'] = inst
        
        return df_daily[['date', 'instrument', 'daily_factor']].rename(columns={'daily_factor': 'factor'})
        
    except Exception as e:
        print(f"Error processing {inst}: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_daily_factor(start_date: str, end_date: str, batch_size: int = 500) -> pd.DataFrame:
    """
    计算日频流动性弹性因子。
    
    Returns:
        DataFrame with columns: ['date', 'instrument', 'factor']
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


def apply_rolling_smoothing(df: pd.DataFrame, 
                            window_days: int = FACTOR_WINDOW,
                            min_periods: int = 10) -> pd.DataFrame:
    """
    对日频因子值进行滚动平滑处理（低频化）。
    """
    if df is None or df.empty:
        print("警告：输入数据为空，无法进行平滑处理")
        return df
    
    print(f"\n开始执行{window_days}个交易日滚动平滑...")
    print(f"平滑前数据量: {len(df)}条")
    
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    df['smoothed_factor'] = df.groupby('instrument')['factor'].transform(
        lambda x: x.rolling(window=window_days, min_periods=min_periods).mean()
    )
    
    df_smoothed = df.dropna(subset=['smoothed_factor']).reset_index(drop=True)
    
    print(f"平滑后数据量: {len(df_smoothed)}条")
    print(f"删除数据量: {len(df) - len(df_smoothed)}条（前{min_periods-1}个交易日无足够数据）")
    
    return df_smoothed


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
    use_rolling_smoothing = True
    
    print(f"=== {FACTOR_NAME} 流动性弹性因子计算 ===")
    print(f"目标表: {FACTOR_LIBRARY_TABLE}（宽表格式）")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日")
    print(f"因子方向: 负向（因子值越小，未来收益越高）")
    print(f"核心方法: 成交量分布集中度（基尼系数）")
    
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
    
    # 计算日频因子
    print("\n步骤1: 计算日频流动性弹性因子...")
    start_time_calc = time.time()
    df_factor = calculate_daily_factor(effective_start, end_date, batch_size=batch_size)
    calc_time = time.time() - start_time_calc
    
    if df_factor is not None and not df_factor.empty:
        print(f"\n日频因子计算完成: {len(df_factor)} 条")
        print(f"日期范围: {df_factor['date'].min().date()} ~ {df_factor['date'].max().date()}")
        print(f"股票数量: {df_factor['instrument'].nunique()}")
        
        # 滚动平滑（低频化）
        if use_rolling_smoothing:
            print("\n步骤2: 滚动平滑低频化...")
            df_factor = apply_rolling_smoothing(df_factor, window_days=FACTOR_WINDOW)
            factor_col = 'smoothed_factor'
        else:
            factor_col = 'factor'
        
        if not df_factor.empty:
            # 过滤到用户指定日期范围
            df_factor['date'] = pd.to_datetime(df_factor['date'])
            original_count = len(df_factor)
            df_factor = df_factor[
                (df_factor['date'] >= pd.to_datetime(start_date)) & 
                (df_factor['date'] <= pd.to_datetime(end_date))
            ].reset_index(drop=True)
            print(f"过滤后: {len(df_factor)} 条（原{original_count}条）")
            
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
                print(f"计算耗时: {calc_time:.2f}秒")
                print(f"写入耗时: {write_time:.2f}秒")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"'{FACTOR_NAME}' 流动性弹性因子共写入 {written_count} 条数据到 '{FACTOR_LIBRARY_TABLE}'")
                print(f"\n预期表现:")
                print(f"  - Rank IC: -7.15%（预估）")
                print(f"  - 核心逻辑: 成交量分布集中度")
                print(f"  - 集中度越高，流动性弹性越大")
            else:
                print("过滤后没有数据需要写入。")
        else:
            print("平滑后没有数据。")
    else:
        print("未能计算任何因子值，流程终止。")

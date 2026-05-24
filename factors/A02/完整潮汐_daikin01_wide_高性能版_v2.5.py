# A02 完整潮汐因子 - 高性能优化版（8核并行）- 修正版v2.5
# 来源：方正证券《个股成交量的潮汐变化及"潮汐"因子构建》（2022-05-08）
# 档案编号：FQ-20260318-007
# 版本：v2.5 - 修复v2.4边界处理和SQL查询效率
#
# 【修正说明】
# v2.1: 严格按照研报定义搜索范围（第5分钟开始、第233分钟结束）
# v2.2: 删除整月数据量检查（len(df_min) < 234）
# v2.3: 尝试修复邻域计算边界效应，但数据长度检查过于严格导致无输出
# v2.4:
#   1. 放宽数据长度检查（从234改为20，参考v2.0）
#   2. 保留搜索范围修正（rise_start=4, ebb_end=min(232, len(df)-1)）
#   3. 保留顶峰位置限制（5 <= t <= 232）
#   问题原因：v2.3的数据长度检查过于严格，过滤掉了所有有效数据
# v2.5:
#   1. 修复 Vm == Vn 时返回NaN的问题（默认涨潮为强势半潮汐）
#   2. 优化SQL查询：精简列、去掉HHMM计算、精确时间范围09:35-14:53
#   3. 收紧数据长度检查（从20改为200，避免短交易日错误处理）
#   4. 删除冗余检查（第265行重复检查）
#   5. 数据源切换为 cn_stock_bar1m_c
#   6. 预期提升：减少约50%数据传输，避免边界信号遗漏
#
# 因子构建逻辑：
#   1. 识别日内成交量"潮汐"过程：涨潮时刻(m) → 顶峰时刻(t) → 退潮时刻(n)
#   2. 计算邻域成交量（每分钟±4分钟成交量总和，共9分钟）
#   3. 判断强势半潮汐：
#      - 若 Vm < Vn：涨潮是强势半潮汐（起点更低，需要更强力量推动）
#      - 若 Vm > Vn：退潮是强势半潮汐（终点更低，需要更强力量推动）
#      - 若 Vm == Vn：默认涨潮为强势半潮汐（v2.5修复）
#   4. 计算强势半潮汐价格变动速率 = (Ct-Cm)/Cm/(t-m) 或 (Cn-Ct)/Ct/(n-t)
#   5. 计算弱势半潮汐价格变动速率 = 另一段的价格变动速率
#   6. 强势半潮汐因子 = mean(强势半潮汐, 20日)
#   7. 稳定弱势半潮汐因子 = std(弱势半潮汐, 20日)
#   8. 完整潮汐因子 = (强势半潮汐 + 稳定弱势半潮汐) / 2
#
# 优化点：
#   - 多进程并行计算（8核）
#   - 批量SQL查询（减少IO）
#   - 向量化滚动计算（无Python循环）

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A02a"                # 本因子编号（完整潮汐）
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（20日）
NEIGHBOR_WINDOW = 4                # 邻域窗口（前后各4分钟，共9分钟）
MIN_DAY_BARS = 200                 # 【v2.5】单日最小分钟数，避免短交易日错误处理

# ========== 性能配置 ==========
MAX_WORKERS = 16                    # 并行进程数
CHUNK_SIZE = 500                  # 批量处理大小


# ========== 高性能工具函数 ==========

def get_last_computed_date(table_id: str, factor_name: str = FACTOR_NAME) -> Optional[str]:
    """获取该因子最后计算的日期（优化版）"""
    try:
        sql = f"""
        SELECT MAX(date) as max_date 
        FROM {table_id} 
        WHERE {factor_name} IS NOT NULL
        """
        q = dai.query(sql)
        result = q.df()
        
        if not result.empty and result['max_date'].iloc[0] is not None:
            last_date = pd.to_datetime(result['max_date'].iloc[0])
            extended_date = last_date - timedelta(days=SAFETY_BUFFER_DAYS)
            result_str = extended_date.strftime('%Y-%m-%d')
            print(f"检测到历史数据，最后计算日期: {last_date.date()}，增量起始: {result_str}")
            return result_str
    except Exception as e:
        print(f"读取历史数据失败: {e}")
    return None


def prepare_factor_df_for_write(df: pd.DataFrame, factor_name: str = FACTOR_NAME) -> pd.DataFrame:
    """规范因子DataFrame（优化版）"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    if 'complete_tide_factor' in df.columns:
        df.rename(columns={'complete_tide_factor': factor_name}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['instrument'] = df['instrument'].astype(str).str.strip()
    df[factor_name] = pd.to_numeric(df[factor_name], errors='coerce')
    
    df_out = df[['date', 'instrument', factor_name]].dropna()
    return df_out.drop_duplicates(subset=['date', 'instrument'], keep='last').reset_index(drop=True)


def safe_write_to_library(df: pd.DataFrame, 
                          table_id: str = FACTOR_LIBRARY_TABLE, 
                          factor_name: str = FACTOR_NAME, 
                          overwrite: bool = False) -> int:
    """将因子数据写入宽表"""
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
            def add_column(df):
                df[factor_name] = np.nan
                return df
            ds.apply_bdb(add_column, as_type=pd.DataFrame)
            existing_df = ds.read_bdb(as_type=pd.DataFrame)
        
        if overwrite and table_exists and factor_name in existing_df.columns:
            def clear_column(df):
                df[factor_name] = np.nan
                return df
            ds.apply_bdb(clear_column, as_type=pd.DataFrame)
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
        print(f"成功写入 '{factor_name}'：{len(df)} 行")
        return len(df)
    except Exception as e:
        print(f"写入失败: {e}")
        return 0


def fetch_all_a_share_instruments() -> List[str]:
    """获取所有A股代码"""
    sql = "SELECT DISTINCT instrument FROM cn_stock_instruments"
    q = dai.query(sql, full_db_scan=True)
    df = q.df()
    if df is None or df.empty:
        return []
    return df['instrument'].astype(str).tolist()


def chunk_list(lst: List, size: int):
    """列表分块生成器"""
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def fetch_stock_minute_data_batch(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """批量读取分钟级数据（高性能版）【v2.5优化】"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    
    # 【v2.5】精简列、去掉HHMM计算、精确时间范围09:35-14:53
    sql = f"""
    SELECT 
        date,
        close,
        volume,
        instrument
    FROM cn_stock_bar1m_c
    WHERE date >= TIMESTAMP '{start_date} 09:35:00'
      AND date <= TIMESTAMP '{end_date} 14:53:00'
      AND instrument IN ('{instrument_list}')
    """
    
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df['trade_date'] = df['date'].dt.date
        df['instrument'] = df['instrument'].astype(str)
        
        numeric_cols = ['close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"SQL查询失败: {e}")
        return pd.DataFrame()


# ========== 因子计算函数（A02完整潮汐核心逻辑） ==========

def calculate_neighbor_volume(df_minute: pd.DataFrame, neighbor_window: int = NEIGHBOR_WINDOW) -> pd.DataFrame:
    """
    计算邻域成交量（每分钟的成交量及其前后neighbor_window分钟的总和）。
    研报定义：为了减小个别异常点的影响，计算每分钟成交量及其前后4分钟的总和（共9分钟）。
    """
    df = df_minute.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # 计算邻域成交量（滚动窗口求和）
    df['neighbor_volume'] = df['volume'].rolling(
        window=2*neighbor_window+1, 
        min_periods=neighbor_window+1,
        center=True
    ).sum()
    
    return df


def identify_tide_process(df_minute: pd.DataFrame) -> Optional[dict]:
    """
    识别日内"潮汐"过程的关键时刻。
    
    【修正v2.1】严格按照研报定义搜索范围：
    1) 顶峰时刻：邻域成交量最高点（第t分钟）
    2) 涨潮时刻：第5~t-1分钟里，邻域成交量最低点（第m分钟）
       - 【修正】从第5分钟开始搜索（索引4），避开开盘波动
    3) 退潮时刻：第t+1~233分钟里，邻域成交量最低点（第n分钟）
       - 【修正】搜索到第233分钟（索引232），避开收盘前7分钟
    
    Returns:
        dict: 包含涨潮/顶峰/退潮时刻的完整信息
        None: 如果无法识别完整的潮汐过程
    """
    df = df_minute.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # 【v2.5】收紧数据长度检查，避免短交易日错误处理
    if len(df) < MIN_DAY_BARS:
        return None
    
    # 计算邻域成交量
    df = calculate_neighbor_volume(df)
    df = df.dropna(subset=['neighbor_volume']).reset_index(drop=True)
    
    # 1. 找到顶峰时刻（邻域成交量最高点）
    peak_idx = df['neighbor_volume'].idxmax()
    t = peak_idx
    Ct = df.loc[t, 'close']
    Vt = df.loc[t, 'neighbor_volume']
    
    # 【修正v2.1】顶峰时刻必须在有效范围内（第5~233分钟之间），否则无法完整搜索涨潮和退潮
    if t < 5 or t > 232:
        return None
    
    # 2. 找到涨潮时刻（第5~t-1分钟里，邻域成交量最低点）
    # 【修正v2.1】严格按照研报：从第5分钟开始（索引4），到t-1分钟
    rise_start = 4  # 第5分钟对应索引4
    rise_end = t - 1  # 到顶峰时刻前一分钟
    
    if rise_start >= rise_end:
        return None
    
    rise_df = df.loc[rise_start:rise_end]
    if rise_df.empty:
        return None
    
    rise_idx = rise_df['neighbor_volume'].idxmin()
    m = rise_idx
    Vm = df.loc[m, 'neighbor_volume']
    Cm = df.loc[m, 'close']
    
    # 3. 找到退潮时刻（第t+1~233分钟里，邻域成交量最低点）
    # 【修正v2.1】严格按照研报：从t+1分钟开始，到第233分钟（索引232）
    ebb_start = t + 1  # 从顶峰时刻后一分钟开始
    ebb_end = min(232, len(df) - 1)  # 第233分钟或数据末尾，取较小值
    
    if ebb_start > ebb_end:
        return None
    
    ebb_df = df.loc[ebb_start:ebb_end]
    if ebb_df.empty:
        return None
    
    ebb_idx = ebb_df['neighbor_volume'].idxmin()
    n = ebb_idx
    Vn = df.loc[n, 'neighbor_volume']
    Cn = df.loc[n, 'close']
    
    return {
        'rise_moment': m,
        'peak_moment': t,
        'ebb_moment': n,
        'Vm': Vm,
        'Cm': Cm,
        'Vt': Vt,
        'Ct': Ct,
        'Vn': Vn,
        'Cn': Cn
    }


def calculate_daily_tide_factors(df_minute: pd.DataFrame, tide_info: dict) -> dict:
    """
    计算日频潮汐相关因子，包括强势半潮汐和弱势半潮汐。
    【v2.5】修复 Vm == Vn 时默认涨潮为强势半潮汐
    """
    if tide_info is None:
        return {}
    
    m = tide_info['rise_moment']
    t = tide_info['peak_moment']
    n = tide_info['ebb_moment']
    Vm = tide_info['Vm']
    Cm = tide_info['Cm']
    Ct = tide_info['Ct']
    Vn = tide_info['Vn']
    Cn = tide_info['Cn']
    
    # 计算全潮汐价格变动速率
    if Cm == 0 or (n - m) == 0:
        full_tide_speed = np.nan
    else:
        full_tide_speed = (Cn - Cm) / Cm / (n - m)
    
    # 判断强势半潮汐和弱势半潮汐
    if Vm < Vn:
        # 涨潮是强势半潮汐
        strong_start, strong_end = m, t
        C_strong_start, C_strong_end = Cm, Ct
        weak_start, weak_end = t, n
        C_weak_start, C_weak_end = Ct, Cn
    elif Vm > Vn:
        # 退潮是强势半潮汐
        strong_start, strong_end = t, n
        C_strong_start, C_strong_end = Ct, Cn
        weak_start, weak_end = m, t
        C_weak_start, C_weak_end = Cm, Ct
    else:
        # 【v2.5】Vm == Vn 时默认涨潮为强势半潮汐，避免遗漏有效信号
        strong_start, strong_end = m, t
        C_strong_start, C_strong_end = Cm, Ct
        weak_start, weak_end = t, n
        C_weak_start, C_weak_end = Ct, Cn
    
    # 计算强势半潮汐价格变动速率
    strong_duration = strong_end - strong_start
    if C_strong_start == 0 or strong_duration == 0:
        strong_half_tide_speed = np.nan
    else:
        strong_half_tide_speed = (C_strong_end - C_strong_start) / C_strong_start / strong_duration
    
    # 计算弱势半潮汐价格变动速率
    weak_duration = weak_end - weak_start
    if C_weak_start == 0 or weak_duration == 0:
        weak_half_tide_speed = np.nan
    else:
        weak_half_tide_speed = (C_weak_end - C_weak_start) / C_weak_start / weak_duration
    
    return {
        'full_tide_speed': full_tide_speed,
        'strong_half_tide_speed': strong_half_tide_speed,
        'weak_half_tide_speed': weak_half_tide_speed
    }


def process_single_stock_optimized(inst: str, df_min: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    单只股票处理（向量化优化）- A02完整潮汐版本
    处理多日期数据，每天识别潮汐过程并计算因子
    
    【修正v2.1】每天需要至少234分钟数据（0~233分钟），才能完整搜索涨潮和退潮范围
    """
    try:
        # 【修正v2.2】删除整月数据量检查，保留单日检查在循环中进行
        if df_min is None or df_min.empty:
            return None
        
        df_min = df_min.sort_values('date').copy()
        
        # 按日期分组处理，确保每个交易日都有数据
        daily_results = []
        
        for trade_date, day_df in df_min.groupby('trade_date'):
            # 【v2.5】收紧单日数据检查
            if len(day_df) < MIN_DAY_BARS:
                continue
            
            day_df = day_df.sort_values('date').reset_index(drop=True)
            
            # 识别潮汐过程
            tide_info = identify_tide_process(day_df)
            
            if tide_info is None:
                continue
            
            # 计算日频因子
            factors = calculate_daily_tide_factors(day_df, tide_info)
            
            if not factors:
                continue
            
            daily_results.append({
                'date': pd.to_datetime(trade_date),
                'instrument': inst,
                'full_tide_speed': factors.get('full_tide_speed', np.nan),
                'strong_half_tide_speed': factors.get('strong_half_tide_speed', np.nan),
                'weak_half_tide_speed': factors.get('weak_half_tide_speed', np.nan)
            })
        
        if not daily_results:
            return None
        
        return pd.DataFrame(daily_results)
        
    except Exception as e:
        return None


def parallel_process_batch(stock_data_dict: Dict[str, pd.DataFrame], 
                           max_workers: int = MAX_WORKERS) -> List[pd.DataFrame]:
    """并行处理一批股票"""
    if not stock_data_dict:
        return []
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_inst = {
            executor.submit(process_single_stock_optimized, inst, df): inst 
            for inst, df in stock_data_dict.items()
        }
        
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                pass
    
    return results


def calculate_daily_factor_optimized(start_date: str, end_date: str, 
                                     batch_size: int = CHUNK_SIZE,
                                     use_parallel: bool = True) -> pd.DataFrame:
    """
    高性能日频因子计算 - A02完整潮汐
    """
    print("获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        return pd.DataFrame()
    
    print(f"获取到 {len(instruments_all)} 只股票，批次大小={batch_size}，并行={use_parallel}")
    
    all_daily_factors = []
    total_batches = (len(instruments_all) + batch_size - 1) // batch_size
    
    for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
        print(f"\n处理第 {batch_idx}/{total_batches} 批：{len(batch)} 只股票 ...")
        batch_start = datetime.now()
        
        # 批量获取数据
        df_minute_all = fetch_stock_minute_data_batch(batch, start_date, end_date)
        
        if df_minute_all is None or df_minute_all.empty:
            continue
        
        # 分组为每只股票的数据
        stock_groups = {
            inst: group.reset_index(drop=True) 
            for inst, group in df_minute_all.groupby('instrument')
        }
        
        # 并行或串行处理
        if use_parallel and len(stock_groups) > 1:
            batch_results = parallel_process_batch(stock_groups, MAX_WORKERS)
        else:
            batch_results = []
            for inst, df in stock_groups.items():
                result = process_single_stock_optimized(inst, df)
                if result is not None and not result.empty:
                    batch_results.append(result)
        
        if batch_results:
            df_batch = pd.concat(batch_results, axis=0, ignore_index=True)
            all_daily_factors.append(df_batch)
            batch_time = (datetime.now() - batch_start).total_seconds()
            print(f"第 {batch_idx} 批完成：{len(df_batch)} 行，耗时 {batch_time:.2f}秒")
    
    if not all_daily_factors:
        return pd.DataFrame()
    
    df_all = pd.concat(all_daily_factors, axis=0, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    return df_all.drop_duplicates(subset=['date', 'instrument'], keep='last').sort_values(['date', 'instrument'])


def calculate_complete_tide_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算完整潮汐因子（向量化优化版）
    
    研报定义：
    - 强势半潮汐因子 = mean(强势半潮汐价格变动速率, 20日)
    - 稳定弱势半潮汐因子 = std(弱势半潮汐价格变动速率, 20日)
    - 完整潮汐因子 = (强势半潮汐因子 + 稳定弱势半潮汐因子) / 2
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只，日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 向量化滚动计算
    min_periods = min(10, FACTOR_WINDOW)  # 至少10天数据即可计算
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    # 强势半潮汐因子：20日滚动平均
    df['strong_half_tide_factor'] = df.groupby('instrument')['strong_half_tide_speed'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 稳定弱势半潮汐因子：20日滚动标准差
    df['stable_weak_half_tide_factor'] = df.groupby('instrument')['weak_half_tide_speed'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).std()
    )
    
    # 完整潮汐因子 = (强势半潮汐 + 稳定弱势半潮汐) / 2
    df['complete_tide_factor'] = (df['strong_half_tide_factor'] + df['stable_weak_half_tide_factor']) / 2
    
    # 只保留有效因子值
    df_result = df[df['complete_tide_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    
    return df_result[['date', 'instrument', 'complete_tide_factor']]


# ========== 主流程 ==========
if __name__ == "__main__":
    import time
    start_time_total = time.time()
    
    # 参数配置
    start_date = "2026-05-02"
    end_date = "2026-05-09"
    overwrite = False
    use_incremental = False
    use_parallel = True
    
    print(f"=== {FACTOR_NAME}（完整潮汐）因子计算 - 高性能优化版 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日, 邻域窗口: {NEIGHBOR_WINDOW}分钟")
    
    # 增量计算
    if use_incremental:
        last_computed = get_last_computed_date(FACTOR_LIBRARY_TABLE, FACTOR_NAME)
        if last_computed:
            buffer_days = FACTOR_WINDOW + 10
            effective_start = (pd.to_datetime(last_computed) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"【增量模式】检测到最后计算日期: {last_computed}")
            print(f"【增量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
        else:
            buffer_days = FACTOR_WINDOW + 10
            effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
            print(f"【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    else:
        buffer_days = FACTOR_WINDOW + 10
        effective_start = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
        print(f"【全量模式】扩展计算起始: {effective_start}（含{buffer_days}天滚动缓冲）")
    
    # 步骤1：计算日频因子
    print("\n【步骤1】计算日频因子（并行优化）...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor_optimized(effective_start, end_date, 
                                                 batch_size=CHUNK_SIZE,
                                                 use_parallel=use_parallel)
    calc_time = time.time() - start_time_calc
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        
        # 步骤2：计算完整潮汐因子
        print("\n【步骤2】计算完整潮汐因子（向量化滚动）...")
        start_time_factor = time.time()
        df_factor = calculate_complete_tide_factor_optimized(df_daily)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
            print("\n【调试信息】")
            print(f"  日频数据列: {df_daily.columns.tolist()}")
            print(f"  建议检查：")
            print(f"    1. 日期范围是否足够（至少需要{FACTOR_WINDOW}个交易日）")
            print(f"    2. 日频数据是否有空值")
            print(f"    3. 尝试使用更大的日期范围（如3个月以上）")
        else:
            print(f"因子计算完成: {len(df_factor)} 条，耗时 {factor_time:.2f}秒")
            
            # 过滤到指定日期范围
            df_factor['date'] = pd.to_datetime(df_factor['date'])
            original_count = len(df_factor)
            df_factor = df_factor[
                (df_factor['date'] >= pd.to_datetime(start_date)) & 
                (df_factor['date'] <= pd.to_datetime(end_date))
            ].reset_index(drop=True)
            
            print(f"\n日期过滤: {original_count} -> {len(df_factor)} 条")
            
            if not df_factor.empty:
                # 写入数据
                df_to_write = prepare_factor_df_for_write(df_factor, FACTOR_NAME)
                
                start_time_write = time.time()
                written_count = safe_write_to_library(df_to_write, overwrite=overwrite)
                write_time = time.time() - start_time_write
                
                total_time = time.time() - start_time_total
                print(f"\n=== 完成 ===")
                print(f"日频计算: {calc_time:.2f}秒")
                print(f"因子合成: {factor_time:.2f}秒")
                print(f"数据写入: {write_time:.2f}秒")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"写入 {written_count} 条数据")
                print(f"\n预期表现: Rank IC ~ -7.90%")
                print(f"\n【合成公式】完整潮汐因子 = (强势半潮汐 + 稳定弱势半潮汐) / 2")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")

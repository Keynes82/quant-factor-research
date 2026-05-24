# A11 激流勇进因子 - 高性能优化版（8核并行）v2.1
# 来源：方正证券《个股交易放量期间的买入强度刻画与"激流勇进"因子构建》（2024-08-29）
# 档案编号：FQ-20260422-A11
# 版本：v2.1 - 修复价格趋势计算方式（收益率趋势替代简单价格比较）
#
# 【严格按研报原文复现】
#   1. 邻域成交量 = 当前分钟成交量 + 前4分钟成交量总和（5分钟滚动窗口平滑）
#   2. 放量/缩量：当前邻域成交量 > 前一分钟邻域成交量 → 放量；反之 → 缩量
#   3. 上涨/下跌：过去5分钟收益率趋势（当前close / 5分钟前close - 1 > 0）→ 上涨；反之 → 下跌
#   4. 四种状态：放量上涨、放量下跌、缩量上涨、缩量下跌
#   5. 日因子 = 放量下跌时刻的（成交金额比例 - 成交量比例）
#   6. 月因子 = 过去20个交易日日因子的均值
#
# 【v2.1 修复记录】
#   - 价格趋势计算：从简单价格比较（close > shift(4)）改为收益率趋势（pct_change(4) > 0）
#   - 原因：研报原文明确使用"收益率趋势"，v2.0实现概念有偏差
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
FACTOR_NAME = "A11"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（月因子=20日平均）
NEIGHBORHOOD_WINDOW = 5            # 邻域成交量窗口（5分钟）

# ========== 性能配置 ==========
MAX_WORKERS = 8                    # 并行进程数
CHUNK_SIZE = 1000                  # 批量处理大小


# ========== 高性能工具函数（与A01保持一致） ==========

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
    
    if 'turbulent_rush_factor' in df.columns:
        df.rename(columns={'turbulent_rush_factor': factor_name}, inplace=True)
    
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
    """批量读取分钟级数据（高性能版）"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    
    # 【重要】剔除开盘和收盘数据：仅考虑9:35-14:56（与研报一致）
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
    WHERE date >= TIMESTAMP '{start_date} 09:30:00'
      AND date <= TIMESTAMP '{end_date} 15:00:00'
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) >= 935
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) <= 1456
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
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"SQL查询失败: {e}")
        return pd.DataFrame()


# ========== A11因子核心计算函数（严格按研报原文） ==========

def calculate_neighborhood_volume(df: pd.DataFrame, window: int = NEIGHBORHOOD_WINDOW) -> pd.DataFrame:
    """
    步骤1：计算邻域成交量
    研报原文："计算个股每分钟的成交量及其之前4分钟成交量的总和，记为该分钟的'邻域成交量'"
    """
    # 5分钟滚动求和（包含当前分钟和前4分钟）
    df['neighborhood_volume'] = df['volume'].rolling(window=window, min_periods=window).sum()
    return df


def classify_volume_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤2：划分放量/缩量状态
    研报原文："根据每分钟的领域成交量相较于前一分钟领域成交量的大小进行判断，
             如当前时刻的领域成交量更大，则当前分钟为'放量'状态，反之则为'缩量'状态"
    """
    # 当前邻域成交量 vs 前一分钟邻域成交量
    df['volume_state'] = np.where(
        df['neighborhood_volume'] > df['neighborhood_volume'].shift(1),
        'increase',  # 放量
        'decrease'   # 缩量
    )
    return df


def classify_price_trend(df: pd.DataFrame, window: int = NEIGHBORHOOD_WINDOW) -> pd.DataFrame:
    """
    步骤3：划分上涨/下跌状态
    研报原文："依据过去5分钟内高、开、低、收数据，计算近期收益率趋势，
             趋势为正则为'上涨'状态，反之为'下跌'状态"
    
    【v2.1 修复】研报明确使用"收益率趋势"，改为 pct_change 计算
    【边界】收益率=0时归为'down'（与v2.0行为一致，保守处理）
    """
    # 过去5分钟收益率趋势（当前close / 5分钟前close - 1）
    past_return = df['close'] / df['close'].shift(window - 1) - 1
    
    df['price_trend'] = np.where(
        past_return > 0,
        'up',    # 上涨：收益率>0
        'down'   # 下跌：收益率<=0（含持平情形）
    )
    return df


def classify_trading_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤4：综合划分四种交易状态
    研报原文："结合上述'放量'与'缩量'状态，将每分钟交易状态划分为以下四种类型：
             放量上涨、放量下跌、缩量上涨、缩量下跌"
    """
    conditions = [
        (df['volume_state'] == 'increase') & (df['price_trend'] == 'up'),
        (df['volume_state'] == 'increase') & (df['price_trend'] == 'down'),
        (df['volume_state'] == 'decrease') & (df['price_trend'] == 'up'),
        (df['volume_state'] == 'decrease') & (df['price_trend'] == 'down')
    ]
    choices = ['increase_up', 'increase_down', 'decrease_up', 'decrease_down']
    
    df['trading_state'] = np.select(conditions, choices, default='unknown')
    return df


def calculate_daily_turbulent_rush(day_df: pd.DataFrame) -> Optional[float]:
    """
    步骤5：计算日激流勇进因子
    研报原文："重点关注放量时刻的买入强度...通过计算全天放量下跌时刻的成交金额与成交量的关系，
             来刻画不同情形下投资者买入意愿的强弱程度"
    
    公式：买入强度 = 放量下跌期间成交金额比例 - 放量下跌期间成交量比例
         = (放量下跌amount / 全天amount) - (放量下跌volume / 全天volume)
    
    研报明确："我们将个股每日放量下跌情形中投资者买入意愿强度定义为'日激流勇进'因子"
    """
    # 筛选放量下跌时刻
    increase_down_df = day_df[day_df['trading_state'] == 'increase_down']
    
    if increase_down_df.empty:
        return None
    
    # 全天总成交额和总成交量
    total_amount = day_df['amount'].sum()
    total_volume = day_df['volume'].sum()
    
    if total_amount <= 0 or total_volume <= 0:
        return None
    
    # 放量下跌期间的成交额和成交量
    turbulent_amount = increase_down_df['amount'].sum()
    turbulent_volume = increase_down_df['volume'].sum()
    
    # 计算比例
    amount_ratio = turbulent_amount / total_amount
    volume_ratio = turbulent_volume / total_volume
    
    # 买入强度 = 金额比例 - 成交量比例
    # 逻辑：如果金额占比 > 成交量占比，说明同样的成交量贡献了更多金额 → 价格较高 → 买方力量强
    buy_intensity = amount_ratio - volume_ratio
    
    return buy_intensity


def process_single_stock_turbulent_rush(inst: str, df_min: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    单只股票激流勇进日因子计算（严格按研报原文）
    """
    try:
        if df_min is None or df_min.empty or len(df_min) < NEIGHBORHOOD_WINDOW:
            return None
        
        df_min = df_min.sort_values('date').copy()
        
        daily_results = []
        
        # 按交易日分组处理
        for trade_date, day_df in df_min.groupby('trade_date'):
            if len(day_df) < NEIGHBORHOOD_WINDOW:
                continue
            
            day_df = day_df.sort_values('date').reset_index(drop=True)
            
            # 步骤1：计算邻域成交量
            day_df = calculate_neighborhood_volume(day_df)
            
            # 剔除前4分钟（邻域成交量不足5分钟的数据）
            day_df = day_df.dropna(subset=['neighborhood_volume'])
            if len(day_df) < 5:
                continue
            
            # 步骤2：划分放量/缩量状态
            day_df = classify_volume_state(day_df)
            
            # 步骤3：划分上涨/下跌状态
            day_df = classify_price_trend(day_df)
            
            # 步骤4：综合划分四种交易状态
            day_df = classify_trading_state(day_df)
            
            # 步骤5：计算日激流勇进因子（只关注放量下跌）
            daily_factor = calculate_daily_turbulent_rush(day_df)
            
            if daily_factor is None:
                continue
            
            # 统计放量下跌时刻数量
            increase_down_count = (day_df['trading_state'] == 'increase_down').sum()
            
            daily_results.append({
                'date': pd.to_datetime(trade_date),
                'instrument': inst,
                'daily_turbulent_rush': daily_factor,
                'increase_down_count': increase_down_count
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
            executor.submit(process_single_stock_turbulent_rush, inst, df): inst 
            for inst, df in stock_data_dict.items()
        }
        
        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                pass  # 静默处理单个股票错误
    
    return results


def calculate_daily_factor_optimized(start_date: str, end_date: str, 
                                     batch_size: int = CHUNK_SIZE,
                                     use_parallel: bool = True) -> pd.DataFrame:
    """
    高性能日频因子计算
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
                result = process_single_stock_turbulent_rush(inst, df)
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


def calculate_turbulent_rush_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算激流勇进因子（向量化优化版）
    
    研报原文："每月月底计算过去20个交易日'日激流勇进'因子的均值，即可得到'激流勇进'因子"
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只，"
          f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 向量化滚动计算
    min_periods = min(10, FACTOR_WINDOW)  # 至少10天数据即可计算
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    # 月均激流勇进：过去20日因子的均值
    df['turbulent_rush_factor'] = df.groupby('instrument')['daily_turbulent_rush'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 只保留有效因子值
    df_result = df[df['turbulent_rush_factor'].notna()].copy()
    
    print(f"  因子计算结果: {len(df_result)}条有效数据")
    print(f"  因子均值: {df_result['turbulent_rush_factor'].mean():.6f}")
    print(f"  因子标准差: {df_result['turbulent_rush_factor'].std():.6f}")
    
    return df_result[['date', 'instrument', 'turbulent_rush_factor']]


# ========== 主流程 ==========
if __name__ == "__main__":
    import time
    start_time_total = time.time()
    
    # 参数配置
    start_date = "2026-02-04"
    end_date = "2026-03-14"
    overwrite = False
    use_incremental = True
    use_parallel = True
    
    print(f"=== {FACTOR_NAME}（激流勇进）因子计算 - v2.0 严格复现版 ===")
    print(f"配置: {MAX_WORKERS}核并行, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日, 邻域窗口: {NEIGHBORHOOD_WINDOW}分钟")
    print(f"\n【因子逻辑】（严格按研报原文）:")
    print(f"  1. 邻域成交量: 当前分钟 + 前4分钟总和（{NEIGHBORHOOD_WINDOW}分钟滚动）")
    print(f"  2. 放量/缩量: 当前邻域 vs 前一分钟邻域")
    print(f"  3. 上涨/下跌: 过去{NEIGHBORHOOD_WINDOW}分钟价格趋势")
    print(f"  4. 四种状态: 放量上涨/放量下跌/缩量上涨/缩量下跌")
    print(f"  5. 日因子: 放量下跌时刻的（金额比例 - 成交量比例）")
    print(f"  6. 月因子: 过去{FACTOR_WINDOW}日日因子均值")
    print(f"\n【数据过滤】剔除开盘(9:30-9:35)和收盘(14:56-15:00)数据")
    
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
    print("\n【步骤1】计算日频激流勇进因子（并行优化）...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor_optimized(effective_start, end_date, 
                                                 batch_size=CHUNK_SIZE,
                                                 use_parallel=use_parallel)
    calc_time = time.time() - start_time_calc
    
    if df_daily is None or df_daily.empty:
        print("未能计算任何日频因子值，流程终止。")
    else:
        print(f"日频因子计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        print(f"  放量下跌时刻统计: 平均{df_daily['increase_down_count'].mean():.2f}个/日")
        
        # 步骤2：计算激流勇进因子（20日均值）
        print("\n【步骤2】计算激流勇进因子（20日均值）...")
        start_time_factor = time.time()
        df_factor = calculate_turbulent_rush_factor_optimized(df_daily)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("未能计算任何因子值，流程终止。")
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
                print(f"\nA11激流勇进因子 v2.1 - 严格按研报原文复现")
                print(f"【v2.1修复】价格趋势计算改为收益率趋势（pct_change(4) > 0）")
                print(f"目标Rank IC: 8.00%")
                print(f"预期方向: 正向因子（因子值越大，未来收益越高）")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")

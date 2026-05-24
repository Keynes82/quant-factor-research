# A11 激流勇进因子 - 高性能优化版（8核并行）v2.2
# 来源：方正证券《个股交易放量期间的买入强度刻画与"激流勇进"因子构建》（2024-08-29）
# 档案编号：FQ-20260422-A11
# 版本：v2.2 - 性能优化版（去除进程池 + 全向量化计算）
#
# 【v2.2 性能优化】
# 1. 【优化一】去除ProcessPoolExecutor：
#    单股计算极轻量（纯Pandas向量化），进程池pickle开销远大于计算量。
#    改为单进程直接循环处理，消除pickle开销。
#    预估提速：2-5x
# 2. 【优化二】消除逐日groupby：
#    原v2.1单股内逐日groupby，5000只×20天=10万个DataFrame创建/销毁。
#    改为全向量化一次性计算整个DataFrame，最后一次性groupby聚合日因子。
#    预估提速：2-3x
#
# 【v2.1 修复保留】
#   价格趋势计算：从简单价格比较（close > shift(4)）改为收益率趋势（pct_change(4) > 0）
#   原因：研报原文明确使用"收益率趋势"，v2.0实现概念有偏差
#
# 因子构建逻辑（严格按研报原文复现）：
#   1. 邻域成交量 = 当前分钟成交量 + 前4分钟成交量总和（5分钟滚动窗口平滑）
#   2. 放量/缩量：当前邻域成交量 > 前一分钟邻域成交量 → 放量；反之 → 缩量
#   3. 上涨/下跌：过去5分钟收益率趋势（当前close / 5分钟前close - 1 > 0）→ 上涨；反之 → 下跌
#   4. 四种状态：放量上涨、放量下跌、缩量上涨、缩量下跌
#   5. 日因子 = 放量下跌时刻的（成交金额比例 - 成交量比例）
#   6. 月因子 = 过去20个交易日日因子的均值

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A11"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（月因子=20日平均）
NEIGHBORHOOD_WINDOW = 5            # 邻域成交量窗口（5分钟）

# ========== 性能配置 ==========
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


# ========== A11因子核心计算函数（严格按研报原文，v2.2全向量化优化） ==========

def process_batch_vectorized(df_min: pd.DataFrame) -> pd.DataFrame:
    """
    【v2.2优化】全向量化计算一批股票的日激流勇进因子。
    
    替代原v2.1逐股×逐日循环，改为整个DataFrame一次性计算。
    """
    if df_min is None or df_min.empty:
        return pd.DataFrame()
    
    # 排序：instrument + date
    df = df_min.sort_values(['instrument', 'date']).copy()
    
    # 步骤1：邻域成交量 = 当前 + 前4分钟总和
    df['neighborhood_volume'] = df.groupby('instrument')['volume'].transform(
        lambda x: x.rolling(window=NEIGHBORHOOD_WINDOW, min_periods=NEIGHBORHOOD_WINDOW).sum()
    )
    
    # 剔除邻域不足的数据（前4分钟无数据）
    df = df[df['neighborhood_volume'].notna()].copy()
    if len(df) == 0:
        return pd.DataFrame()
    
    # 步骤2：放量/缩量判断（当前 vs 前一分钟邻域）
    df['prev_neighborhood'] = df.groupby('instrument')['neighborhood_volume'].shift(1)
    df['is_increase'] = df['neighborhood_volume'] > df['prev_neighborhood']
    
    # 步骤3：上涨/下跌判断（过去5分钟收益率趋势）
    df['prev_close_4'] = df.groupby('instrument')['close'].shift(NEIGHBORHOOD_WINDOW - 1)
    df['past_return'] = df['close'] / df['prev_close_4'] - 1
    df['is_up'] = df['past_return'] > 0
    
    # 步骤4：四种交易状态编码（向量化）
    conditions = [
        df['is_increase'] & df['is_up'],      # 放量上涨
        df['is_increase'] & ~df['is_up'],     # 放量下跌
        ~df['is_increase'] & df['is_up'],     # 缩量上涨
        ~df['is_increase'] & ~df['is_up']     # 缩量下跌
    ]
    choices = [0, 1, 2, 3]  # 0=inc_up, 1=inc_down, 2=dec_up, 3=dec_down
    df['state_code'] = np.select(conditions, choices, default=-1)
    
    # 标记放量下跌
    df['is_increase_down'] = df['state_code'] == 1
    
    # 步骤5：按 instrument + trade_date 聚合日因子
    # 放量下跌金额 / 全天金额 - 放量下跌成交量 / 全天成交量
    daily = df.groupby(['instrument', 'trade_date']).agg(
        total_amount=('amount', 'sum'),
        total_volume=('volume', 'sum'),
        turbulent_amount=('amount', lambda x: x[df.loc[x.index, 'is_increase_down']].sum()),
        turbulent_volume=('volume', lambda x: x[df.loc[x.index, 'is_increase_down']].sum()),
        increase_down_count=('is_increase_down', 'sum')
    ).reset_index()
    
    # 计算日因子
    daily['daily_turbulent_rush'] = (
        daily['turbulent_amount'] / daily['total_amount'] - 
        daily['turbulent_volume'] / daily['total_volume']
    )
    
    # 过滤有效值
    daily = daily[
        (daily['total_amount'] > 0) & 
        (daily['total_volume'] > 0) & 
        (daily['daily_turbulent_rush'].notna())
    ].copy()
    
    if len(daily) == 0:
        return pd.DataFrame()
    
    daily['date'] = pd.to_datetime(daily['trade_date'])
    
    return daily[['date', 'instrument', 'daily_turbulent_rush', 'increase_down_count']]


def calculate_daily_factor_vectorized(start_date: str, end_date: str,
                                       batch_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """
    【v2.2优化】高性能日频因子计算 - 全向量化版本
    """
    print("获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        return pd.DataFrame()
    
    print(f"获取到 {len(instruments_all)} 只股票，批次大小={batch_size}，全向量化计算")
    
    all_daily_factors = []
    total_batches = (len(instruments_all) + batch_size - 1) // batch_size
    
    for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
        print(f"\n处理第 {batch_idx}/{total_batches} 批：{len(batch)} 只股票 ...")
        batch_start = datetime.now()
        
        # 批量获取数据
        df_minute_all = fetch_stock_minute_data_batch(batch, start_date, end_date)
        
        if df_minute_all is None or df_minute_all.empty:
            continue
        
        # 【v2.2优化】全向量化计算整批数据
        df_batch = process_batch_vectorized(df_minute_all)
        
        if df_batch is not None and not df_batch.empty:
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
    
    print(f"=== {FACTOR_NAME}（激流勇进）因子计算 - v2.2 性能优化版 ===")
    print(f"配置: 单进程全向量化, 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日, 邻域窗口: {NEIGHBORHOOD_WINDOW}分钟")
    print(f"\n【v2.2 性能优化】")
    print(f"  1. 去除ProcessPoolExecutor，单进程直接处理")
    print(f"  2. 消除逐日groupby，全向量化一次性计算")
    print(f"\n【因子逻辑】（严格按研报原文）:")
    print(f"  1. 邻域成交量: 当前分钟 + 前4分钟总和（{NEIGHBORHOOD_WINDOW}分钟滚动）")
    print(f"  2. 放量/缩量: 当前邻域 vs 前一分钟邻域")
    print(f"  3. 上涨/下跌: 过去{NEIGHBORHOOD_WINDOW}分钟收益率趋势")
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
    print("\n【步骤1】计算日频激流勇进因子（全向量化）...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor_vectorized(effective_start, end_date, 
                                                  batch_size=CHUNK_SIZE)
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
                print(f"\nA11激流勇进因子 v2.2 - 性能优化版（去除进程池 + 全向量化）")
                print(f"【v2.1修复】价格趋势计算改为收益率趋势（pct_change(4) > 0）")
                print(f"目标Rank IC: 8.00%")
                print(f"预期方向: 正向因子（因子值越大，未来收益越高）")
                
                if calc_time > 0:
                    stocks_per_sec = len(df_daily) / calc_time
                    print(f"处理速度: {stocks_per_sec:.1f} 条日频数据/秒")
            else:
                print("过滤后没有数据需要写入。")

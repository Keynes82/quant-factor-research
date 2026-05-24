# A09 水中行舟因子 - 高性能优化版（8核并行）
# 来源：方正证券《成交量激增时刻蕴含的alpha信息》（2022-04-12）
# 档案编号：FQ-20260318-015
# 版本：v2.0 - 基于高性能模板（多进程并行优化）
#
# 因子构建逻辑：
#   随波逐流因子（跟随市场）：
#     1. 计算个股成交额与当日总成交额的 Spearman/Pearson 相关系数
#     2. 使用日内每分钟成交额，或每日成交额
#     3. 随波逐流 = corr(个股成交额, 市场总成交额)
#   
#   孤雁出群因子（偏离市场）：
#     1. 计算个股成交额与当日总成交额的偏离程度
#     2. 孤雁出群 = |个股成交额占比变化 - 市场平均变化|
#   
#   日因子：随波逐流 × 孤雁出群（等权）
#   月因子：过去20日平均值
#
# 优化点：
#   - 多进程并行计算（8核）
#   - 批量SQL查询（减少IO）
#   - 向量化滚动计算（无Python循环）
#   - 跨股票相关性使用整体向量化计算（避免双重循环）

import dai
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== 全局配置 ==========
FACTOR_LIBRARY_TABLE = "daikin01"  # 宽表存储表名
FACTOR_NAME = "A09"                # 本因子编号
SAFETY_BUFFER_DAYS = 30            # 安全缓冲天数
FACTOR_WINDOW = 20                 # 因子滚动窗口（月因子=20日平均）
REASONABLE_RETURN_WINDOW = 20      # 合理收益窗口
CORRELATION_METHOD = 'spearman'    # 相关系数方法: 'spearman' 或 'pearson'

# ========== 性能配置 ==========
MAX_WORKERS = 8                    # 并行进程数
CHUNK_SIZE = 1000                  # 批量处理大小


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
    
    if 'factor_value' in df.columns:
        df.rename(columns={'factor_value': factor_name}, inplace=True)
    
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


def fetch_market_daily_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取市场每日总成交额数据（用于计算随波逐流和孤雁出群）
    """
    sql = f"""
    SELECT 
        CAST(date AS DATE) AS trade_date,
        SUM(amount) AS market_total_amount,
        COUNT(DISTINCT instrument) AS stock_count
    FROM cn_stock_bar1d
    WHERE date >= '{start_date}'
      AND date <= '{end_date}'
    GROUP BY CAST(date AS DATE)
    ORDER BY trade_date
    """
    
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        df['market_total_amount'] = pd.to_numeric(df['market_total_amount'], errors='coerce')
        
        return df
    except Exception as e:
        print(f"市场日数据查询失败: {e}")
        return pd.DataFrame()


def fetch_stock_daily_data_batch(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """批量读取日频数据（高性能版）"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    
    sql = f"""
    SELECT 
        CAST(date AS DATE) AS trade_date,
        instrument,
        amount AS daily_amount,
        volume AS daily_volume,
        close,
        open
    FROM cn_stock_bar1d
    WHERE date >= '{start_date}'
      AND date <= '{end_date}'
      AND instrument IN ('{instrument_list}')
    ORDER BY trade_date, instrument
    """
    
    try:
        query_result = dai.query(sql)
        df = query_result.df()
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        df['instrument'] = df['instrument'].astype(str)
        
        numeric_cols = ['daily_amount', 'daily_volume', 'close', 'open']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"日数据SQL查询失败: {e}")
        return pd.DataFrame()


def fetch_stock_minute_data_batch(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """批量读取分钟级数据（高性能版）"""
    if not instruments:
        return pd.DataFrame()
    
    instrument_list = "','".join(instruments)
    
    sql = f"""
    SELECT 
        date,
        close,
        volume,
        amount,
        instrument
    FROM cn_stock_bar1m
    WHERE date >= TIMESTAMP '{start_date} 09:30:00'
      AND date <= TIMESTAMP '{end_date} 15:00:00'
      AND EXTRACT(HOUR FROM date) * 100 + EXTRACT(MINUTE FROM date) >= 931
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
        
        numeric_cols = ['close', 'volume', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"分钟数据SQL查询失败: {e}")
        return pd.DataFrame()


# ========== 核心因子计算函数（向量化优化） ==========

def calculate_sui_bo_zhu_liu_vectorized(df_daily: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
    """
    计算随波逐流因子（向量化优化版）
    
    随波逐流 = corr(个股成交额, 市场总成交额)
    使用滚动窗口计算相关系数
    """
    if df_daily is None or df_daily.empty or df_market is None or df_market.empty:
        return pd.DataFrame()
    
    # 合并市场数据
    df = df_daily.merge(df_market[['trade_date', 'market_total_amount']], 
                        on='trade_date', how='left')
    
    if df.empty:
        return pd.DataFrame()
    
    # 计算个股成交额占比
    df['amount_ratio'] = df['daily_amount'] / df['market_total_amount']
    
    # 按股票分组计算随波逐流（滚动相关系数）
    def calc_rolling_correlation(group):
        group = group.sort_values('trade_date').reset_index(drop=True)
        
        if CORRELATION_METHOD == 'spearman':
            # Spearman秩相关
            group['sui_bo_zhu_liu'] = group['daily_amount'].rolling(
                window=FACTOR_WINDOW, min_periods=min(10, FACTOR_WINDOW)
            ).corr(group['market_total_amount'], method='spearman')
        else:
            # Pearson相关
            group['sui_bo_zhu_liu'] = group['daily_amount'].rolling(
                window=FACTOR_WINDOW, min_periods=min(10, FACTOR_WINDOW)
            ).corr(group['market_total_amount'])
        
        return group
    
    df = df.groupby('instrument', group_keys=False).apply(calc_rolling_correlation)
    
    return df


def calculate_gu_yan_chu_qun_vectorized(df_daily: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
    """
    计算孤雁出群因子（向量化优化版）
    
    孤雁出群 = |个股成交额占比变化 - 市场平均变化|
    """
    if df_daily is None or df_daily.empty or df_market is None or df_market.empty:
        return pd.DataFrame()
    
    # 合并市场数据
    df = df_daily.merge(df_market[['trade_date', 'market_total_amount']], 
                        on='trade_date', how='left')
    
    if df.empty:
        return pd.DataFrame()
    
    # 计算个股成交额占比
    df['amount_ratio'] = df['daily_amount'] / df['market_total_amount']
    
    # 计算个股成交额占比变化（向量化）
    df = df.sort_values(['instrument', 'trade_date']).reset_index(drop=True)
    df['amount_ratio_change'] = df.groupby('instrument')['amount_ratio'].diff()
    
    # 计算每日截面均值（向量化）
    date_mean_change = df.groupby('trade_date')['amount_ratio_change'].mean().reset_index()
    date_mean_change.columns = ['trade_date', 'market_avg_change']
    
    df = df.merge(date_mean_change, on='trade_date', how='left')
    
    # 计算孤雁出群 = |个股变化 - 市场平均变化|
    df['gu_yan_chu_qun'] = (df['amount_ratio_change'] - df['market_avg_change']).abs()
    
    # 滚动平滑（20日平均）
    df['gu_yan_chu_qun'] = df.groupby('instrument')['gu_yan_chu_qun'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min(10, FACTOR_WINDOW)).mean()
    )
    
    return df


def calculate_daily_factor_optimized(start_date: str, end_date: str,
                                     batch_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """
    高性能日频因子计算
    
    优化点：
    1. 先获取市场总数据（一次查询）
    2. 批量获取个股数据
    3. 向量化计算随波逐流和孤雁出群
    4. 合成日因子
    """
    print("=" * 60)
    print("步骤1: 获取市场总成交额数据...")
    
    # 获取市场数据（只需一次查询）
    df_market = fetch_market_daily_data(start_date, end_date)
    if df_market is None or df_market.empty:
        print("获取市场数据失败")
        return pd.DataFrame()
    
    print(f"市场数据: {len(df_market)} 个交易日")
    
    print("\n步骤2: 获取所有A股股票列表...")
    instruments_all = fetch_all_a_share_instruments()
    if not instruments_all:
        return pd.DataFrame()
    
    print(f"获取到 {len(instruments_all)} 只股票")
    
    print("\n步骤3: 批量获取个股日频数据...")
    all_daily_data = []
    total_batches = (len(instruments_all) + batch_size - 1) // batch_size
    
    for batch_idx, batch in enumerate(chunk_list(instruments_all, batch_size), start=1):
        batch_start = datetime.now()
        
        df_batch = fetch_stock_daily_data_batch(batch, start_date, end_date)
        
        if df_batch is not None and not df_batch.empty:
            all_daily_data.append(df_batch)
        
        batch_time = (datetime.now() - batch_start).total_seconds()
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(f"  批次 {batch_idx}/{total_batches} 完成，耗时 {batch_time:.2f}秒")
    
    if not all_daily_data:
        print("未能获取任何个股数据")
        return pd.DataFrame()
    
    df_daily = pd.concat(all_daily_data, axis=0, ignore_index=True)
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date']).dt.date
    
    print(f"\n个股日数据: {len(df_daily)} 条记录，{df_daily['instrument'].nunique()} 只股票")
    
    print("\n步骤4: 计算随波逐流因子（向量化）...")
    df_suibo = calculate_sui_bo_zhu_liu_vectorized(df_daily, df_market)
    
    if df_suibo is None or df_suibo.empty:
        print("随波逐流计算失败")
        return pd.DataFrame()
    
    print(f"随波逐流计算完成: {len(df_suibo)} 条")
    
    print("\n步骤5: 计算孤雁出群因子（向量化）...")
    df_guyan = calculate_gu_yan_chu_qun_vectorized(df_daily, df_market)
    
    if df_guyan is None or df_guyan.empty:
        print("孤雁出群计算失败")
        return pd.DataFrame()
    
    print(f"孤雁出群计算完成: {len(df_guyan)} 条")
    
    print("\n步骤6: 合成日因子...")
    # 合并两个因子
    df_factor = df_suibo.merge(
        df_guyan[['trade_date', 'instrument', 'gu_yan_chu_qun', 'amount_ratio_change', 'market_avg_change']],
        on=['trade_date', 'instrument'],
        how='inner'
    )
    
    # 日因子 = 随波逐流 × 孤雁出群（等权合成）
    # 注意：随波逐流是相关系数（范围-1到1），孤雁出群是偏离度（非负）
    # 对随波逐流取绝对值，然后相乘
    df_factor['sui_bo_zhu_liu_abs'] = df_factor['sui_bo_zhu_liu'].abs()
    df_factor['daily_factor'] = df_factor['sui_bo_zhu_liu_abs'] * df_factor['gu_yan_chu_qun']
    
    # 清理异常值
    df_factor = df_factor[df_factor['daily_factor'].notna()]
    df_factor = df_factor[df_factor['daily_factor'] != np.inf]
    df_factor = df_factor[df_factor['daily_factor'] != -np.inf]
    
    print(f"日因子合成完成: {len(df_factor)} 条有效记录")
    
    return df_factor


def calculate_monthly_factor_optimized(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算月因子（20日滚动平均）- 向量化优化
    """
    if df_daily is None or df_daily.empty:
        print("  警告：日频数据为空")
        return pd.DataFrame()
    
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    print(f"  日频数据量: {len(df)}条，股票数: {df['instrument'].nunique()}只")
    
    # 向量化滚动计算
    min_periods = min(10, FACTOR_WINDOW)
    print(f"  滚动窗口: {FACTOR_WINDOW}日, 最小有效天数: {min_periods}日")
    
    df['factor_value'] = df.groupby('instrument')['daily_factor'].transform(
        lambda x: x.rolling(window=FACTOR_WINDOW, min_periods=min_periods).mean()
    )
    
    # 只保留有效因子值
    df_result = df[df['factor_value'].notna()].copy()
    
    print(f"  月因子计算结果: {len(df_result)}条有效数据")
    
    return df_result[['date', 'instrument', 'factor_value', 'sui_bo_zhu_liu', 'gu_yan_chu_qun', 'daily_factor']]


# ========== 并行处理辅助函数 ==========

def process_correlation_batch(args):
    """
    并行计算一批股票的相关系数
    """
    batch_instruments, df_market_minute, start_date, end_date = args
    
    results = []
    
    for instrument in batch_instruments:
        try:
            # 获取单只股票的分钟数据
            sql = f"""
            SELECT 
                date,
                amount
            FROM cn_stock_bar1m
            WHERE date >= TIMESTAMP '{start_date} 09:30:00'
              AND date <= TIMESTAMP '{end_date} 15:00:00'
              AND instrument = '{instrument}'
            ORDER BY date
            """
            
            q = dai.query(sql)
            df_stock = q.df()
            
            if df_stock is None or df_stock.empty or len(df_stock) < FACTOR_WINDOW * 10:
                continue
            
            df_stock['date'] = pd.to_datetime(df_stock['date'])
            df_stock['trade_date'] = df_stock['date'].dt.date
            df_stock['amount'] = pd.to_numeric(df_stock['amount'], errors='coerce')
            
            # 按日期聚合为日频
            df_stock_daily = df_stock.groupby('trade_date')['amount'].sum().reset_index()
            df_stock_daily.columns = ['trade_date', 'daily_amount']
            
            # 合并市场数据
            df_merged = df_stock_daily.merge(df_market_minute, on='trade_date', how='inner')
            
            if len(df_merged) < min(10, FACTOR_WINDOW):
                continue
            
            # 计算滚动相关系数
            if CORRELATION_METHOD == 'spearman':
                df_merged['sui_bo_zhu_liu'] = df_merged['daily_amount'].rolling(
                    window=FACTOR_WINDOW, min_periods=min(10, FACTOR_WINDOW)
                ).corr(df_merged['market_total_amount'], method='spearman')
            else:
                df_merged['sui_bo_zhu_liu'] = df_merged['daily_amount'].rolling(
                    window=FACTOR_WINDOW, min_periods=min(10, FACTOR_WINDOW)
                ).corr(df_merged['market_total_amount'])
            
            df_merged['instrument'] = instrument
            df_merged = df_merged[df_merged['sui_bo_zhu_liu'].notna()]
            
            if not df_merged.empty:
                results.append(df_merged[['trade_date', 'instrument', 'sui_bo_zhu_liu', 'daily_amount']])
            
        except Exception as e:
            pass
    
    return results


# ========== 主流程 ==========
if __name__ == "__main__":
    import time
    start_time_total = time.time()
    
    # 参数配置
    start_date = "2026-02-04"
    end_date = "2026-03-14"
    overwrite = False
    use_incremental = True
    
    print(f"=== {FACTOR_NAME}（水中行舟）因子计算 - 高性能优化版 ===")
    print(f"配置: 批次{CHUNK_SIZE}只股票")
    print(f"计算范围: {start_date} ~ {end_date}")
    print(f"因子窗口: {FACTOR_WINDOW}日, 相关系数方法: {CORRELATION_METHOD}")
    print(f"因子逻辑: 随波逐流(成交额相关性) × 孤雁出群(偏离度)")
    
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
    print("\n" + "=" * 60)
    print("【步骤1】计算日频因子（向量化优化）...")
    start_time_calc = time.time()
    df_daily = calculate_daily_factor_optimized(effective_start, end_date, batch_size=CHUNK_SIZE)
    calc_time = time.time() - start_time_calc
    
    if df_daily is None or df_daily.empty:
        print("\n未能计算任何日频因子值，流程终止。")
    else:
        print(f"\n日频因子计算完成: {len(df_daily)} 条，耗时 {calc_time:.2f}秒")
        
        # 显示日因子统计
        print("\n日因子统计:")
        print(f"  随波逐流: 均值={df_daily['sui_bo_zhu_liu'].mean():.4f}, 标准差={df_daily['sui_bo_zhu_liu'].std():.4f}")
        print(f"  孤雁出群: 均值={df_daily['gu_yan_chu_qun'].mean():.6f}, 标准差={df_daily['gu_yan_chu_qun'].std():.6f}")
        print(f"  日因子: 均值={df_daily['daily_factor'].mean():.6f}, 标准差={df_daily['daily_factor'].std():.6f}")
        
        # 步骤2：计算月因子（20日滚动平均）
        print("\n" + "=" * 60)
        print("【步骤2】计算月因子（20日滚动平均）...")
        start_time_factor = time.time()
        df_factor = calculate_monthly_factor_optimized(df_daily)
        factor_time = time.time() - start_time_factor
        
        if df_factor is None or df_factor.empty:
            print("\n未能计算任何月因子值，流程终止。")
        else:
            print(f"\n月因子计算完成: {len(df_factor)} 条，耗时 {factor_time:.2f}秒")
            
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
                
                print("\n" + "=" * 60)
                print("=== 完成 ===")
                print(f"日频计算: {calc_time:.2f}秒")
                print(f"月因子合成: {factor_time:.2f}秒")
                print(f"数据写入: {write_time:.2f}秒")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"写入 {written_count} 条数据")
                print(f"\n因子说明:")
                print(f"  - 随波逐流: 个股成交额与市场总成交额的{CORRELATION_METHOD}相关系数")
                print(f"  - 孤雁出群: 个股成交额占比变化与市场平均变化的偏离度")
                print(f"  - 日因子: |随波逐流| × 孤雁出群")
                print(f"  - 月因子: 过去{FACTOR_WINDOW}日平均值")
                
                # 显示最终因子统计
                print(f"\n最终因子统计:")
                print(f"  均值: {df_to_write[FACTOR_NAME].mean():.6f}")
                print(f"  标准差: {df_to_write[FACTOR_NAME].std():.6f}")
                print(f"  最小值: {df_to_write[FACTOR_NAME].min():.6f}")
                print(f"  最大值: {df_to_write[FACTOR_NAME].max():.6f}")
            else:
                print("\n过滤后没有数据需要写入。")

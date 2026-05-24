# BigQuant 个人投资者交易数据探测脚本
# 逐个尝试查询可能存在的数据表

import dai

# 探测列表：可能包含逐笔/资金流数据的表名或字段
table_checks = [
    # 逐笔成交相关
    ("cn_stock_tick", "逐笔成交数据"),
    ("cn_stock_transaction", "成交明细"),
    ("cn_stock_deal", "交易详情"),
    ("cn_stock_detail", "明细数据"),
    ("cn_stock_tickdata", "tick数据"),
    # 资金流相关
    ("cn_stock_moneyflow", "资金流"),
    ("cn_stock_capital_flow", "资金流向"),
    ("cn_stock_fund_flow", "资金流"),
    ("cn_stock_cashflow", "现金流"),
    # 其他可能
    ("cn_stock_order", "委托数据"),
    ("cn_stock_l2", "Level2数据"),
]

print("=" * 60)
print("BigQuant 逐笔成交/资金流数据探测")
print("=" * 60)

for table_name, desc in table_checks:
    try:
        sql = f"SELECT * FROM {table_name} LIMIT 1"
        result = dai.query(sql)
        df = result.df()
        if df is not None and not df.empty:
            print(f"\n✅ [{table_name}] - {desc} - 存在!")
            print(f"   字段: {list(df.columns)}")
            # 检查是否有金额相关字段
            amount_keywords = ['amount', 'money', 'volume', 'price', 'vol', 'value', 'turnover']
            found = [c for c in df.columns if any(k in c.lower() for k in amount_keywords)]
            if found:
                print(f"   疑似金额/成交量字段: {found}")
    except Exception as e:
        # 表不存在或无权访问
        pass

# 尝试通过information_schema查看数据表
print("\n" + "=" * 60)
print("尝试获取所有表名（information_schema方式）")
print("=" * 60)
try:
    sql = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
      AND table_name LIKE '%tick%'
    LIMIT 20
    """
    result = dai.query(sql)
    df = result.df()
    if df is not None and not df.empty:
        print(df)
except Exception as e:
    print(f"information_schema查询失败: {e}")

try:
    sql = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
      AND table_name LIKE '%money%'
    LIMIT 20
    """
    result = dai.query(sql)
    df = result.df()
    if df is not None and not df.empty:
        print(df)
except Exception as e:
    print(f"information_schema查询失败: {e}")

try:
    sql = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
      AND table_name LIKE '%flow%'
    LIMIT 20
    """
    result = dai.query(sql)
    df = result.df()
    if df is not None and not df.empty:
        print(df)
except Exception as e:
    print(f"information_schema查询失败: {e}")

# 如果上面都找不到，尝试在bar1d/bar1m中找turnover相关字段
print("\n" + "=" * 60)
print("检查日频表(bar1d)是否有额外字段")
print("=" * 60)
try:
    sql = "SELECT * FROM cn_stock_bar1d LIMIT 1"
    result = dai.query(sql)
    df = result.df()
    if df is not None and not df.empty:
        print(f"cn_stock_bar1d 字段: {list(df.columns)}")
except Exception as e:
    print(f"查询失败: {e}")

print("\n" + "=" * 60)
print("检查分钟表(bar1m)是否有额外字段")
print("=" * 60)
try:
    sql = "SELECT * FROM cn_stock_bar1m LIMIT 1"
    result = dai.query(sql)
    df = result.df()
    if df is not None and not df.empty:
        print(f"cn_stock_bar1m 字段: {list(df.columns)}")
except Exception as e:
    print(f"查询失败: {e}")

print("\n探测完成。")

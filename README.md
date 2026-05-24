# A-share Quantitative Factor Research Library

A股量化因子研究库，包含因子挖掘、回测验证、绩效分析等完整工作流。

## 因子目录

| 因子代码 | 因子名称 | 最新版本 | 状态 |
|---------|---------|---------|------|
| A01 | 待补充 | - | 研究中 |
| A03 | 球队硬币 | v3.7 | 已入库 |
| A04 | 多空博弈 | v2.1 | 已入库 |
| A05 | 云开雾散 | v2.1 | 已入库 |
| A06 | 飞蛾扑火 | v2.1 | 已入库 |

## 回测规范

- 买入：每周第一个交易日开盘价
- 卖出：每周最后一个交易日收盘价
- 本金：20万
- 持仓：5只
- 佣金：万2
- 回测期：20220101 - 20251231

## 数据规范

- 因子数据：daikin01 宽表
- 行情数据：cn_stock_bar1d

## 快速开始

```bash
# 克隆仓库
git clone git@github.com:Keynes82/quant-factor-research.git
cd quant-factor-research

# 查看因子列表
ls factors/
```

# EATA-RL: Enhanced Adaptive Trading Agent with Reinforcement Learning

[![GitHub](https://img.shields.io/github/license/JNU-Tangyin/eata)](https://github.com/JNU-Tangyin/eata)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## 🚀 项目简介

EATA-RL是一个基于强化学习的增强自适应交易智能体，结合了蒙特卡洛树搜索(MCTS)、神经网络和多种技术指标，用于股票交易决策。本项目经过重构和优化，具有清晰的模块化架构和完整的实验验证系统。

### ✨ 主要特性
- 🧠 **智能决策**: MCTS + 神经网络 + 强化学习三重融合
- 🔬 **科学验证**: 完整的消融实验和对比实验系统  
- 📊 **性能优异**: 多项指标优于传统交易算法
- 🛠️ **易于扩展**: 模块化设计，支持新算法集成
- 📈 **实时交易**: 支持定时调度的生产环境部署

## 📁 项目结构

```
EATA-RL-main/
├── core/                           # 核心模块
│   ├── agent.py                   # 主要交易智能体
│   ├── data.py                    # 数据处理模块
│   ├── env.py                     # 交易环境
│   ├── globals.py                 # 全局配置
│   ├── performance_metrics.py     # 性能指标计算
│   ├── utils.py                   # 工具函数
│   └── eata_agent/               # EATA核心算法
│       ├── args.py               # 参数配置
│       ├── engine.py             # 核心引擎
│       ├── mcts.py               # 蒙特卡洛树搜索
│       ├── model.py              # 模型定义
│       ├── network.py            # 神经网络
│       └── utils/                # 工具模块
├── experiments/                   # 实验模块
│   ├── ablation_study/           # 消融实验
│   │   ├── variants/             # 6个消融变体
│   │   ├── configs/              # 实验配置
│   │   └── run_ablation_study.py # 消融实验主程序
│   └── comparison_experiments/   # 对比实验
│       └── algorithms/           # 对比算法实现
├── docs/                         # 文档
├── predict.py                    # 主要预测接口
├── main.py                      # 定时调度系统
└── experiment_pipeline.py       # 实验流水线
```

## 🔬 核心功能

### 1. EATA核心算法
- **蒙特卡洛树搜索(MCTS)**: 智能决策树搜索
- **神经网络指导**: 深度学习价值评估
- **强化学习反馈**: 自适应学习机制
- **多技术指标融合**: RSI、MACD、布林带等

### 2. 消融实验系统
支持6种消融变体的科学对比：
- **EATA-Full**: 完整版本(基准)
- **EATA-NoNN**: 无神经网络版本
- **EATA-NoMem**: 无记忆机制版本
- **EATA-Simple**: 简化版本
- **EATA-NoExplore**: 无探索机制版本
- **EATA-NoMCTS**: 无蒙特卡洛模拟版本

### 3. 对比实验系统
包含多种主流交易算法：
- **EATA**: 本项目核心算法
- **Buy & Hold**: 买入持有策略
- **LSTM**: 长短期记忆网络
- **ARIMA**: 自回归移动平均
- **GBDT**: 梯度提升决策树
- **LightGBM**: 轻量级梯度提升
- **Transformer**: 注意力机制模型
- **FinRL系列**: PPO、A2C、SAC等强化学习算法

## 🛠️ 安装与使用

### 环境要求
```bash
Python >= 3.8
PyTorch >= 1.9.0
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
```

### 安装依赖
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
pip install quantstats empyrical stockstats baostock tushare
pip install schedule retrying pysnooper streamlit
```

### 快速开始

#### 1. 运行消融实验
```bash
cd experiments/ablation_study
python run_ablation_study.py
```

#### 2. 运行单次预测
```python
from predict import run_eata_core_backtest
import pandas as pd

# 加载股票数据
stock_df = pd.read_csv('your_stock_data.csv')

# 运行EATA回测
metrics, portfolio = run_eata_core_backtest(
    stock_df=stock_df,
    ticker='AAPL',
    lookback=50,
    lookahead=10,
    stride=1,
    depth=300
)

print(f"年化收益率: {metrics['Annual Return (AR)']:.2%}")
print(f"夏普比率: {metrics['Sharpe Ratio']:.2f}")
```

#### 3. 启动定时交易系统
```bash
python main.py
```

## 📊 实验结果

### 消融实验结果示例
| 变体 | 年化收益率 | 夏普比率 | 最大回撤 | 胜率 | 核心特征 |
|------|------------|----------|----------|------|----------|
| EATA-Full | 15.2% | 1.34 | -8.5% | 58.3% | 完整版本(基准) |
| EATA-NoNN | 12.1% | 1.12 | -11.2% | 54.1% | 无神经网络指导 |
| EATA-NoMem | 13.8% | 1.25 | -9.8% | 56.7% | 无记忆机制 |
| EATA-Simple | 14.1% | 1.28 | -9.2% | 57.1% | 简化版本 |
| EATA-NoExplore | 13.5% | 1.21 | -10.1% | 55.8% | 无探索机制 |
| EATA-NoMCTS | 11.9% | 1.08 | -12.8% | 53.2% | 无蒙特卡洛搜索 |

### 对比实验结果示例
| 算法 | 年化收益率 | 夏普比率 | 最大回撤 | 算法类型 |
|------|------------|----------|----------|----------|
| **EATA** | **15.2%** | **1.34** | **-8.5%** | 强化学习+MCTS |
| LSTM | 11.8% | 1.05 | -12.3% | 深度学习 |
| ARIMA | 9.2% | 0.92 | -14.1% | 时间序列 |
| GBDT | 10.5% | 0.98 | -13.5% | 机器学习 |
| Buy&Hold | 8.9% | 0.87 | -15.6% | 基准策略 |

> **注意**: 以上结果为示例数据，实际结果可能因市场环境、参数设置等因素而异。

## 🔧 配置说明

### 主要参数
- `lookback`: 历史数据回看窗口 (默认: 50)
- `lookahead`: 未来预测窗口 (默认: 10)  
- `stride`: 滑动步长 (默认: 1)
- `depth`: MCTS搜索深度 (默认: 300)

### 数据格式
支持标准OHLCV格式：
```csv
date,open,high,low,close,volume,amount
2023-01-01,100.0,105.0,98.0,103.0,1000000,103000000
```

## 📈 性能特点

- **高精度预测**: 结合多种技术指标和深度学习
- **自适应学习**: 强化学习机制持续优化
- **风险控制**: 内置止损和资金管理
- **可扩展性**: 模块化设计，易于扩展新算法
- **科学验证**: 完整的消融实验和对比实验系统

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

如有问题或建议，请通过Issue联系我们。

---

**注意**: 本项目仅用于学术研究和教育目的，不构成投资建议。实际交易请谨慎评估风险。

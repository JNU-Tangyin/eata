# EATA-RL: Enhanced Automated Technical Analysis with Reinforcement Learning

基于强化学习的增强自动化技术分析系统，用于股票交易策略优化。

---

## 📋 项目简介

EATA-RL 是一个结合了蒙特卡洛树搜索（MCTS）和深度强化学习的智能交易系统。通过神经网络引导的搜索算法，自动发现和优化技术分析策略，实现高效的股票交易决策。

### 核心特性

- 🧠 **神经网络引导的MCTS** - 结合深度学习和树搜索的混合算法
- 📊 **完整的消融实验** - 6个变体系统性验证各组件的有效性
- 🔬 **全面的对比实验** - 与FinRL、传统策略、机器学习方法的对比
- 📈 **自动化论文生成** - 统一的后处理脚本生成图表和LaTeX表格
- 🎯 **规范的项目结构** - 数据、实验、结果、论文分离明确

---

## 🏗️ 项目结构

```
EATA-RL-main/
│
├── 📂 core/                          # 核心代码
│   ├── eata_agent/                   # EATA智能体实现
│   │   ├── mcts.py                   # 蒙特卡洛树搜索算法
│   │   ├── model.py                  # 神经网络模型
│   │   ├── agent.py                  # 智能体主类
│   │   ├── engine.py                 # 执行引擎
│   │   └── args.py                   # 参数配置
│   ├── environment/                  # 交易环境
│   └── utils/                        # 工具函数
│
├── 📂 experiments/                   # 实验代码
│   ├── ablation_study/               # 消融实验
│   │   ├── run_ablation_study.py    # 消融实验运行脚本
│   │   ├── configs/                  # 实验配置
│   │   │   ├── experiment_settings.py  # 数据和结果路径配置
│   │   │   └── ablation_config.py      # 消融实验配置
│   │   └── variants/                 # 6个消融变体
│   │       ├── eata_full.py         # Full - 完整模型（基准）
│   │       ├── eata_nonn.py         # NoNN - 无神经网络引导
│   │       ├── eata_nomcts.py       # NoMCTS - 无蒙特卡洛树搜索
│   │       ├── eata_nomem.py        # NoMem - 无经验回放
│   │       ├── eata_noexplore.py    # NoExplore - 无探索机制
│   │       └── eata_simple.py       # Simple - 简化版本
│   │
│   └── comparison_experiments/       # 对比实验
│       └── algorithms/               # 对比算法实现
│           ├── baseline.py          # 统一实验运行入口
│           ├── post.py              # 对比实验后处理
│           ├── data_utils.py        # 数据加载工具
│           ├── eata.py              # EATA算法
│           ├── finrl_strategies.py  # FinRL强化学习方法（PPO/A2C/SAC/TD3/DDPG）
│           ├── lstm.py              # LSTM基线
│           ├── transformer.py       # Transformer基线
│           ├── arima.py             # ARIMA基线
│           ├── gbdt_strategy.py     # GBDT策略
│           ├── buy_and_hold.py      # 买入持有策略
│           └── macd.py              # MACD策略
│
├── 📂 data/                          # 数据目录
│   └── *.csv                        # 20支股票的历史数据
│       # AAPL, AMD, AMT, BA, BAC, BHP, CAT, COST, DE, EQIX,
│       # GE, GOOG, JNJ, JPM, KO, MSFT, NFLX, NVDA, SCHW, XOM
│
├── 📂 results/                       # 实验结果（统一目录）
│   ├── ablation_study/               # 消融实验结果
│   │   ├── raw_results/             # 原始JSON结果
│   │   ├── csv_results/             # CSV汇总
│   │   └── processed_results/       # 处理后的结果
│   └── comparison_study/             # 对比实验结果
│       ├── raw_results/             # 原始实验结果
│       ├── figures/                 # 生成的图表
│       ├── tables/                  # 生成的表格
│       └── detailed_outputs/        # 详细交易数据
│
├── 📂 paper/                         # 论文相关文件
│   ├── 0.main.tex                   # LaTeX主文件
│   ├── figures/                     # 论文图表（PDF/PNG）
│   ├── tables/                      # LaTeX表格
│   └── README.md                    # 论文目录说明
│
├── 📄 post.py                        # 统一后处理脚本
├── 📄 sliding_window_nemots.py      # 滑动窗口EATA核心算法
├── 📄 backtest.py                   # 回测工具
├── 📄 evaluate.py                   # 评估工具
├── 📄 predict.py                    # 预测工具
├── 📄 preprocess.py                 # 数据预处理工具
├── 📄 requirements.txt              # Python依赖
└── 📄 README.md                     # 本文件
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd EATA-RL-main

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

项目已包含20支股票的历史数据，位于 `data/` 目录：
- 数据格式：CSV文件（Date, Open, High, Low, Close, Volume）
- 数据来源：分散的20支股票，覆盖不同行业

### 3. 运行实验

#### 消融实验（6个变体 × 20支股票）

```bash
cd experiments/ablation_study
python run_ablation_study.py
```

**实验配置**：
- 变体数量：6个（Full, NoNN, NoMCTS, NoMem, NoExplore, Simple）
- 股票数量：20支
- 总实验数：120个
- 预计时间：30-32小时

**结果保存位置**：
- 原始结果：`results/ablation_study/raw_results/`
- CSV汇总：`results/ablation_study/csv_results/`

#### 对比实验（多种基线方法）

```bash
cd experiments/comparison_experiments/algorithms
python baseline.py
```

**支持的策略**：
- **传统策略**：Buy & Hold, MACD
- **机器学习**：ARIMA, GBDT, LSTM, Transformer
- **强化学习**：EATA, FinRL (PPO/A2C/SAC/TD3/DDPG)

**结果保存位置**：
- 原始结果：`results/comparison_study/raw_results/`

---

## 📊 生成论文图表

### 使用统一后处理脚本

```bash
# 生成所有图表和表格
python post.py

# 输出位置：
# - 图表：paper/figures/
# - 表格：paper/tables/
```

### 生成的内容

**消融实验**：
- 性能对比图（PDF/PNG）
- 结果表格（LaTeX）
- 股票性能热力图

**对比实验**：
- 策略性能对比图
- 详细统计表格
- 资产曲线图

---

## 📈 实验结果

### 消融实验主要发现

| 变体 | 描述 | 年化收益 | 夏普比率 | 最大回撤 |
|------|------|---------|---------|---------|
| **EATA-Full** | 完整模型 | ✅ 最高 | ✅ 最高 | ✅ 最低 |
| EATA-NoNN | 无神经网络 | ⬇️ 下降 | ⬇️ 下降 | ⬆️ 上升 |
| EATA-NoMCTS | 无MCTS | ⬇️ 显著下降 | ⬇️ 显著下降 | ⬆️ 显著上升 |
| EATA-NoMem | 无经验回放 | ⬇️ 下降 | ⬇️ 下降 | ⬆️ 上升 |
| EATA-NoExplore | 无探索 | ⬇️ 下降 | ⬇️ 下降 | ⬆️ 上升 |
| EATA-Simple | 简化版 | ⬇️ 显著下降 | ⬇️ 显著下降 | ⬆️ 显著上升 |

**结论**：所有组件对系统性能都有显著贡献，验证了EATA架构的有效性。

### 对比实验主要发现

EATA在多个指标上优于传统方法和其他强化学习方法：
- 年化收益率超过Buy & Hold基准
- 夏普比率高于FinRL方法
- 最大回撤控制优于传统技术指标策略

---

## 🔧 配置说明

### 数据路径配置

**消融实验** (`experiments/ablation_study/configs/experiment_settings.py`):
```python
DATA_PATHS = {
    'real_stock_data_dir': PROJECT_ROOT / "data"
}
```

**对比实验** (`experiments/comparison_experiments/algorithms/data_utils.py`):
```python
project_root = Path(__file__).resolve().parents[3]
csv_path = project_root / "data" / f"{ticker}.csv"
```

### 结果路径配置

所有实验结果统一保存到 `results/` 目录：
- 消融实验：`results/ablation_study/`
- 对比实验：`results/comparison_study/`

### 论文输出配置

图表和表格统一保存到 `paper/` 目录：
- 图表：`paper/figures/`
- 表格：`paper/tables/`

---

## 📝 论文编译

```bash
cd paper
pdflatex 0.main.tex
```

论文主文件 `0.main.tex` 会自动引用 `figures/` 和 `tables/` 中的内容。

---

## 🛠️ 核心算法

### EATA架构

```
输入：股票历史数据
  ↓
滑动窗口处理
  ↓
MCTS搜索 ←→ 神经网络引导
  ↓
策略选择（买入/持有/卖出）
  ↓
经验回放 → 网络训练
  ↓
输出：交易决策序列
```

### 关键参数

```python
EATA_DEFAULT_PARAMS = {
    'lookback': 50,          # 回看窗口
    'lookahead': 10,         # 预测窗口
    'stride': 1,             # 步长
    'depth': 300,            # MCTS搜索深度
    'max_len': 35,           # 最大表达式长度
    'exploration_rate': 1 / (2**0.5),  # 探索率
    'lr': 1e-5,              # 学习率
}
```

---

## 📚 依赖项

主要依赖：
- Python >= 3.8
- PyTorch >= 1.10
- pandas, numpy
- matplotlib, seaborn
- FinRL (用于对比实验)
- stable-baselines3 (用于FinRL)

详见 `requirements.txt`

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📄 许可证

详见 `LICENSE` 文件。

---

## 📧 联系方式

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

## 🎯 项目状态

- ✅ 核心算法实现完成
- ✅ 消融实验完成（6变体 × 20股票）
- ✅ 对比实验框架完成
- ✅ 论文图表生成系统完成
- 🔄 持续优化和改进中

---

**最后更新**: 2026年2月24日

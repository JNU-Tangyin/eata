# FinRL和InvestorBench原版集成指南

## 📋 概述

本指南介绍如何在EATA项目中使用**原版FinRL和InvestorBench框架**进行对比实验，复现FLAG-TRADER论文中的实验设置。

⚠️ **重要说明**: 本集成**严格使用原版框架**，不使用任何模拟或简化版本，确保实验结果的真实性和可信度。

### 🎯 集成目标

1. **FinRL原版集成**: 使用真正的FinRL框架和Stable Baselines3算法 (PPO, A2C, SAC, TD3, DDPG)
2. **InvestorBench原版集成**: 使用真实的LLM API调用 (GPT-3.5, GPT-4等)
3. **FLAG-TRADER复现**: 提供与FLAG-TRADER论文相似的实验设置和评估框架
4. **原版验证**: 提供验证脚本确保使用的是原版框架而非模拟版本

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装所有依赖
python install_finrl_investorbench.py --mode all

# 或分别安装
python install_finrl_investorbench.py --mode finrl
python install_finrl_investorbench.py --mode investorbench

# 创建requirements文件
python install_finrl_investorbench.py --create-requirements
```

### 2. 配置环境

#### FinRL配置
- 无需额外配置，适配器会自动处理
- 建议在GPU环境下运行以获得更好性能

#### InvestorBench配置
```bash
# 设置OpenAI API Key
export OPENAI_API_KEY='your-api-key-here'

# 设置HuggingFace缓存目录 (可选)
export HF_HOME='/path/to/huggingface/cache'
```

### 3. 验证原版集成

```bash
# 验证所有组件是否使用原版框架
python verify_authentic_integration.py --check-all

# 只验证FinRL
python verify_authentic_integration.py --check-finrl

# 只验证InvestorBench
python verify_authentic_integration.py --check-investorbench

# 包含API测试的完整验证
python verify_authentic_integration.py --check-all --test-api
```

### 4. 运行测试

```bash
# 测试FinRL集成
python comparison_experiments/algorithms/finrl.py

# 测试InvestorBench集成
python comparison_experiments/algorithms/investorbench.py
```

## 🧪 运行对比实验

### 方式1: 使用FLAG-TRADER风格实验运行器 (推荐)

```bash
# 运行学术论文标准对比实验
python run_flag_trader_experiments.py --experiment_type academic

# 专注于FinRL方法的对比
python run_flag_trader_experiments.py --experiment_type finrl_focus

# 专注于LLM方法的对比
python run_flag_trader_experiments.py --experiment_type llm_focus

# 完整对比实验
python run_flag_trader_experiments.py --experiment_type full

# 自定义股票和参数
python run_flag_trader_experiments.py \
    --experiment_type academic \
    --tickers AAPL MSFT GOOGL \
    --num_runs 5 \
    --lookback 30 \
    --lookahead 5
```

### 方式2: 使用现有实验框架

```bash
# 运行单个参数集实验
python run_experiments.py --mode single \
    --strategies eata finrl_ppo finrl_sac investorbench_gpt35 \
    --tickers AAPL MSFT GOOGL

# 运行参数扫描实验
python run_experiments.py --mode sweep \
    --strategies eata finrl_ppo investorbench_gpt35
```

### 方式3: 直接使用BaselineRunner

```python
from comparison_experiments.algorithms.baseline import BaselineRunner

# 创建运行器
runner = BaselineRunner()

# 选择要测试的策略
strategies = [
    'eata',                    # 我们的方法
    'finrl_ppo', 'finrl_sac',  # FinRL方法
    'investorbench_gpt35',     # LLM方法
    'transformer', 'lstm',     # 传统深度学习
    'buy_and_hold', 'macd'     # 传统基线
]

# 运行实验
results = runner.run_all_strategies(
    df=your_stock_data,
    ticker='AAPL',
    selected_strategies=strategies
)
```

## 📊 实验配置

### 预定义实验类型

1. **academic**: 学术论文标准对比
   - EATA vs FinRL-PPO vs FinRL-SAC vs GPT-3.5 vs Transformer vs LSTM vs Buy&Hold vs MACD

2. **finrl_focus**: FinRL专项对比
   - EATA vs 所有FinRL算法 vs 传统PPO

3. **llm_focus**: LLM专项对比
   - EATA vs 所有InvestorBench LLM vs Transformer vs LSTM

4. **full**: 完整对比
   - 包含所有可用的算法

### 股票集合

- **tech_growth**: 科技成长股 (AAPL, MSFT, GOOGL, AMZN, TSLA)
- **finance**: 金融股 (JPM, BAC, WFC, GS, MS)
- **diverse**: 多样化组合 (AAPL, JPM, JNJ, XOM, WMT)

## 📈 支持的算法

### EATA算法
- `eata`: 我们提出的方法

### FinRL强化学习算法
- `finrl_ppo`: Proximal Policy Optimization
- `finrl_a2c`: Advantage Actor-Critic
- `finrl_sac`: Soft Actor-Critic
- `finrl_td3`: Twin Delayed Deep Deterministic Policy Gradient
- `finrl_ddpg`: Deep Deterministic Policy Gradient

### InvestorBench LLM算法
- `investorbench_gpt35`: GPT-3.5 Turbo
- `investorbench_gpt4`: GPT-4
- `investorbench_llama2`: Llama2-7B
- `investorbench_finbert`: FinBERT

### 传统基线算法
- `buy_and_hold`: 买入持有策略
- `macd`: MACD技术指标策略
- `transformer`: Transformer深度学习模型
- `lstm`: LSTM神经网络
- `lightgbm`: LightGBM机器学习
- `arima`: ARIMA时间序列模型
- `ppo`: 传统PPO实现
- `gp`: 遗传编程

## 📋 评估指标

所有算法使用统一的评估指标：

### 收益指标
- **年化收益率** (Annualized Return): 投资组合的年化收益
- **总收益率** (Total Return): 整个测试期间的总收益
- **超额收益** (Excess Return): 相对于基准的超额收益

### 风险指标
- **夏普比率** (Sharpe Ratio): 风险调整后收益
- **最大回撤** (Max Drawdown): 最大资产损失
- **波动率** (Volatility): 收益率标准差
- **Calmar比率**: 年化收益/最大回撤
- **Sortino比率**: 下行风险调整收益

### 交易指标
- **胜率** (Win Rate): 盈利交易占比
- **交易次数** (Number of Trades): 总交易次数
- **平均持仓时间**: 平均持有期

### LLM特有指标 (InvestorBench)
- **预测准确率** (Prediction Accuracy): LLM预测的准确性
- **平均置信度** (Average Confidence): LLM预测的平均置信度
- **任务完成率** (Task Completion Rate): 成功完成的任务比例

## 📊 结果分析

### 1. 生成实验报告

```bash
# 生成学术论文级别的报告
python experiment_pipeline.py --mode all

# 只生成图表
python experiment_pipeline.py --mode figures

# 只生成LaTeX表格
python experiment_pipeline.py --mode tables
```

### 2. 结果文件结构

```
flag_trader_results/
├── flag_trader_results_academic_20231205_143022.json  # 实验结果
├── experiment_raw_data_20231205_143022.csv           # 原始数据
├── strategy_summary_20231205_143022.csv              # 策略汇总
└── ...

figures/
├── strategy_performance_comparison_20231205_143022.pdf
├── risk_return_scatter_20231205_143022.pdf
├── performance_distribution_20231205_143022.pdf
└── strategy_correlation_20231205_143022.pdf

tables/
├── strategy_performance_20231205_143022.tex
├── detailed_statistics_20231205_143022.tex
└── top_strategies_20231205_143022.tex
```

### 3. 关键分析维度

#### 算法类型对比
- **符号回归** (EATA): 可解释性强，表达式简洁
- **强化学习** (FinRL): 适应性强，在线学习能力
- **大语言模型** (InvestorBench): 多模态信息融合，常识推理
- **传统方法**: 计算效率高，稳定性好

#### 市场环境适应性
- **上涨市场**: 各算法表现差异
- **下跌市场**: 风险控制能力对比
- **震荡市场**: 信号识别准确性

#### 计算效率对比
- **训练时间**: 不同算法的训练耗时
- **推理速度**: 实时预测的响应时间
- **资源消耗**: 内存和GPU使用情况

## 🔧 高级配置

### 1. 自定义FinRL参数

```python
# 在运行实验时传入FinRL特定参数
python run_flag_trader_experiments.py \
    --experiment_type finrl_focus \
    --total_timesteps 100000 \
    --lookback 30
```

### 2. 自定义InvestorBench任务

```python
# 修改investorbench_adapter.py中的任务类型
task_types = [
    'stock_movement_prediction',
    'portfolio_optimization',
    'risk_assessment',
    'market_sentiment_analysis',
    'trading_signal_generation'
]
```

### 3. 添加新的评估指标

```python
# 在baseline.py中扩展评估指标
def calculate_custom_metrics(returns):
    # 添加自定义指标计算
    pass
```

## 🚨 注意事项

### 1. 依赖安装
- FinRL需要较多依赖，建议使用虚拟环境
- InvestorBench的大模型需要足够的内存和存储空间
- OpenAI API需要有效的API Key和足够的配额

### 2. 计算资源
- FinRL训练建议使用GPU加速
- 大型LLM推理需要大量内存
- 完整实验可能需要数小时到数天时间

### 3. 数据质量
- 确保股票数据的完整性和准确性
- 注意处理缺失值和异常值
- 考虑股票分割、分红等公司行为的影响

### 4. 实验可重复性
- 设置随机种子确保结果可重复
- 记录实验参数和环境配置
- 多次运行取平均值减少随机性影响

## 📚 参考资源

### 论文
- **FLAG-TRADER**: "Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading"
- **FinRL**: "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance"
- **InvestorBench**: "InvestorBench: A Comprehensive Benchmark for Financial LLM Evaluation"

### 代码仓库
- [FinRL GitHub](https://github.com/AI4Finance-Foundation/FinRL)
- [InvestorBench GitHub](https://github.com/AI4Finance-Foundation/InvestorBench)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

### 文档
- [FinRL Documentation](https://finrl.readthedocs.io/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## 🤝 贡献指南

如果你想为这个集成项目做出贡献：

1. **报告问题**: 在GitHub Issues中报告bug或提出改进建议
2. **添加新算法**: 参考现有适配器的结构添加新的基线算法
3. **改进评估**: 扩展评估指标或改进实验设计
4. **优化性能**: 提高算法运行效率或减少资源消耗

## 📞 支持

如果在使用过程中遇到问题：

1. 查看本指南的常见问题部分
2. 检查依赖安装是否正确
3. 查看实验日志中的错误信息
4. 参考相关项目的官方文档

---

**祝你实验顺利！🎉**

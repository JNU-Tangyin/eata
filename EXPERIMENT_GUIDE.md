# 学术论文实验系统使用指南

## 🎯 系统概述

本系统提供了完整的学术论文实验数据处理流水线，从原始实验运行到最终的图表和表格生成。

## 📁 目录结构

```
EATA-RL-main/
├── experiment_results/     # 实验原始数据 (CSV格式)
├── figures/               # 论文图表 (PDF + PNG)
├── tables/                # LaTeX表格 (TEX格式)
├── data/                  # 处理后的数据
├── run_experiments.py     # 实验运行脚本
├── experiment_pipeline.py # 数据处理流水线
└── generate_paper_outputs.py # 一键生成脚本
```

## 🚀 快速开始

### 一键生成所有论文输出

```bash
python generate_paper_outputs.py
```

这个命令会：
1. 检查并运行必要的实验
2. 处理实验数据
3. 生成所有图表和表格
4. 输出到 `figures/` 和 `tables/` 目录

## 🔧 详细使用方法

### 1. 运行实验

#### 单参数集实验
```bash
python run_experiments.py --mode single \
    --lookback 50 --lookahead 10 --stride 1 --depth 300 \
    --tickers AAPL MSFT GOOGL \
    --strategies eata buy_and_hold macd transformer \
    --runs 5
```

#### 参数扫描实验
```bash
python run_experiments.py --mode sweep \
    --tickers AAPL MSFT GOOGL AMZN TSLA \
    --strategies eata buy_and_hold macd transformer ppo \
    --runs 3
```

#### 生成汇总
```bash
python run_experiments.py --mode summary
```

### 2. 处理数据和生成输出

#### 生成所有图表和表格
```bash
python experiment_pipeline.py --mode all
```

#### 只生成图表
```bash
python experiment_pipeline.py --mode figures
```

#### 只生成表格
```bash
python experiment_pipeline.py --mode tables
```

#### 只处理数据
```bash
python experiment_pipeline.py --mode data
```

## 📊 输出文件说明

### 实验数据文件 (experiment_results/)

文件命名格式：
```
experiment_results_lookback{lb}_lookahead{la}_stride{s}_depth{d}_{ticker}_{timestamp}.csv
```

例如：
```
experiment_results_lookback50_lookahead10_stride1_depth300_AAPL_20251201_120000.csv
```

CSV文件包含列：
- `run_id`: 运行轮次
- `ticker`: 股票代码
- `strategy`: 策略名称
- `timestamp`: 时间戳
- `lookback`, `lookahead`, `stride`, `depth`: 参数
- `Annual Return (AR)`: 年化收益率
- `Sharpe Ratio`: 夏普比率
- `Max Drawdown (MDD)`: 最大回撤
- `Win Rate`: 胜率
- `Volatility (Annual)`: 年化波动率
- 其他性能指标...

### 图表文件 (figures/)

生成的图表包括：
1. `strategy_performance_comparison_{timestamp}.pdf/png` - 策略性能对比
2. `risk_return_scatter_{timestamp}.pdf/png` - 风险收益散点图
3. `performance_distribution_{timestamp}.pdf/png` - 性能分布箱线图
4. `strategy_correlation_{timestamp}.pdf/png` - 策略相关性热力图

### 表格文件 (tables/)

生成的LaTeX表格包括：
1. `strategy_performance_{timestamp}.tex` - 策略性能汇总表
2. `detailed_statistics_{timestamp}.tex` - 详细统计表
3. `top_strategies_{timestamp}.tex` - 前5名策略对比表

## 📈 实验参数配置

### 默认参数网格
```python
param_grid = {
    'lookback': [30, 50, 100],      # 回望窗口
    'lookahead': [5, 10, 20],       # 预测窗口  
    'stride': [1, 2],               # 步长
    'depth': [200, 300, 500]        # MCTS深度
}
```

### 支持的策略
- `eata`: EATA算法
- `buy_and_hold`: 买入持有
- `macd`: MACD技术指标
- `transformer`: Transformer模型
- `ppo`: PPO强化学习
- `gp`: 遗传编程
- `lstm`: LSTM神经网络
- `lightgbm`: LightGBM机器学习
- `arima`: ARIMA时间序列

## 🎨 图表样式

- 使用ggplot2风格的seaborn主题
- 支持中文字体显示
- 输出高分辨率PDF和PNG格式
- 适合学术论文使用

## 📝 LaTeX表格

生成的表格包含：
- 标准的booktabs样式
- 自动的标题和标签
- 格式化的数值显示
- 可直接插入LaTeX文档

## 🔍 故障排除

### 常见问题

1. **缺少依赖包**
   ```bash
   pip install -r requirements_paper.txt
   ```

2. **实验数据不存在**
   - 先运行 `python run_experiments.py --mode single`
   - 或使用 `python generate_paper_outputs.py` 自动处理

3. **中文字体显示问题**
   - 确保系统安装了中文字体
   - 或修改 `experiment_pipeline.py` 中的字体设置

4. **内存不足**
   - 减少并行实验数量
   - 使用更小的参数网格

### 调试模式

在脚本中添加详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 扩展功能

### 自定义参数网格
```python
custom_grid = {
    'lookback': [40, 60],
    'lookahead': [8, 12], 
    'stride': [1],
    'depth': [250, 350]
}

runner.run_parameter_sweep(custom_grid=custom_grid)
```

### 添加新策略
1. 在 `comparison_experiments/algorithms/` 中实现策略
2. 在 `baseline.py` 的 `STRATEGY_CONFIGS` 中注册
3. 更新 `run_experiments.py` 中的策略列表

### 自定义图表
修改 `experiment_pipeline.py` 中的绘图函数，添加新的可视化类型。

## 📞 技术支持

如有问题，请检查：
1. Python版本 >= 3.8
2. 所有依赖包已安装
3. 数据文件路径正确
4. 足够的磁盘空间

---

**最后更新**: 2025年12月1日

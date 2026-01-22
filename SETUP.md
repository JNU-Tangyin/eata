# EATA-RL 环境设置指南

## 快速开始

### 1. 创建虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  # Windows
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行实验
```bash
cd comparison_experiments/algorithms
python baseline.py AAPL --strategies "eata,buy_and_hold,macd"
```

## 环境要求
- Python 3.9+
- 至少4GB内存
- 推荐使用macOS或Linux

## 常见问题

### 死锁问题
如果遇到死锁，请设置环境变量：
```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```
注意：线程数设置为2可以平衡性能和稳定性。如果仍有问题，可以设置为1。

### GPU支持
- macOS: 自动使用MPS (Apple Silicon)
- Linux: 需要CUDA环境

## 策略说明
- **EATA**: 核心算法，需要完整依赖
- **FinRL**: 强化学习策略
- **传统策略**: Buy&Hold, MACD等
- **深度学习**: LSTM, Transformer

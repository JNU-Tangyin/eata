# EATA-RL Project Structure

## 📁 Project Overview
Enhanced Adaptive Trading Agent with Reinforcement Learning - A sophisticated algorithmic trading system that combines neural networks, Monte Carlo Tree Search (MCTS), and evolutionary algorithms for stock market prediction and trading.

## 🏗️ Directory Structure

```
EATA-RL/
├── 📂 core/                    # Core system components
│   ├── agent.py               # Main trading agent implementation
│   ├── env.py                 # Trading environment
│   ├── data.py                # Data processing utilities
│   ├── globals.py             # Global configurations
│   ├── utils.py               # Utility functions
│   ├── performance_metrics.py # Performance evaluation metrics
│   └── eata_agent/            # EATA-specific agent components
│       ├── engine.py          # Training engine
│       ├── model.py           # Neural network models
│       ├── network.py         # PVNet implementation
│       └── ...
│
├── 📂 experiments/            # All experimental frameworks
│   ├── ablation_study/        # Ablation study experiments
│   │   ├── run_ablation_study.py
│   │   ├── variants/          # Different model variants
│   │   ├── configs/           # Configuration files
│   │   └── results/           # Experimental results
│   ├── comparison_experiments/ # Algorithm comparison studies
│   └── comparison_results/    # Comparison results
│
├── 📂 docs/                   # Documentation
│   ├── README.md              # Project overview
│   ├── EXPERIMENT_GUIDE.md    # Experiment setup guide
│   ├── FINRL_INVESTORBENCH_GUIDE.md
│   └── SETUP.md               # Installation guide
│
├── 📂 data/                   # Data storage
├── 📂 figures/                # Generated plots and visualizations
├── 📂 tables/                 # Result tables
│
├── 📄 main.py                 # Main entry point
├── 📄 predict.py              # Prediction pipeline
├── 📄 backtest.py             # Backtesting utilities
├── 📄 evaluate.py             # Model evaluation
├── 📄 preprocess.py           # Data preprocessing
├── 📄 visualize.py            # Visualization tools
├── 📄 requirements.txt        # Dependencies
└── 📄 .gitignore              # Git ignore rules
```

## 🎯 Key Components

### Core System (`core/`)
- **agent.py**: Main trading agent with MCTS and neural network integration
- **eata_agent/**: Enhanced agent components with evolutionary algorithms
- **env.py**: Trading environment simulation
- **data.py**: Data loading and processing utilities

### Experiments (`experiments/`)
- **ablation_study/**: Systematic component removal studies
  - 6 variants: Full, NoNN, NoMem, Simple, NoExplore, NoMCTS
  - 20 stock dataset for comprehensive evaluation
- **comparison_experiments/**: Baseline algorithm comparisons

### Documentation (`docs/`)
- Complete setup and usage guides
- Experimental protocols and methodologies
- Performance benchmarking results

## 🚀 Quick Start

1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Ablation Study**:
   ```bash
   cd experiments/ablation_study
   python run_ablation_study.py
   ```

3. **Single Stock Prediction**:
   ```bash
   python main.py
   ```

## 📊 Experimental Framework

The project supports multiple experimental setups:
- **Ablation Studies**: Component-wise performance analysis
- **Baseline Comparisons**: Against traditional ML/RL methods
- **Multi-stock Validation**: 20 diverse stocks for robust evaluation
- **Performance Metrics**: Sharpe ratio, annual return, max drawdown, win rate

## 🔧 Configuration

All configurations are centralized in:
- `experiments/ablation_study/configs/` for ablation studies
- Individual experiment scripts for specific setups

## 📈 Results

Results are automatically saved in structured formats:
- CSV files for quantitative metrics
- JSON files for detailed experiment logs
- Markdown reports for human-readable summaries

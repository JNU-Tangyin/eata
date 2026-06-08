"""
EATA消融实验通用设置
包含数据路径、模型参数、评估配置等
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # 指向EATA-RL-main根目录
ABLATION_ROOT = PROJECT_ROOT / "experiments" / "ablation_study"

# 数据路径配置
DATA_PATHS = {
    'raw_data_dir': ABLATION_ROOT / "data" / "raw_data",
    'experiment_data_dir': ABLATION_ROOT / "data" / "experiment_data", 
    'processed_data_dir': ABLATION_ROOT / "data" / "processed_data",
    'stock_data_20': ABLATION_ROOT / "data" / "raw_data" / "stock_data_20.csv",
    'market_indices': ABLATION_ROOT / "data" / "raw_data" / "market_indices.csv",
    # 真实股票数据路径
    'real_stock_data_dir': PROJECT_ROOT / "data"
}

# 结果路径配置 - 统一存放到项目根目录的results/
RESULT_PATHS = {
    'raw_results_dir': PROJECT_ROOT / "results" / "ablation_study" / "raw_results",
    'csv_results_dir': PROJECT_ROOT / "results" / "ablation_study" / "csv_results",
    'processed_results_dir': PROJECT_ROOT / "results" / "ablation_study" / "processed_results",
    'figures_dir': PROJECT_ROOT / "paper" / "figures"  # 图表直接保存到paper/figures/
}

# EATA模型默认参数 - 使用与对比实验完全一致的参数
EATA_DEFAULT_PARAMS = {
    'lookback': 50,  # 与对比实验一致
    'lookahead': 10,  # 与对比实验一致
    'stride': 1,  # 与对比实验一致
    'depth': 300,  # 与对比实验一致
    'max_len': 35,  # 保持修复后的值
    'num_runs': 5,
    'exploration_rate': 1 / (2**0.5),
    'num_transplant': 5,
    'num_aug': 3,
    'eta': 1.0,
    'lr': 1e-5,
    'weight_decay': 0.0001
}

# 实验运行配置
EXPERIMENT_CONFIG = {
    'num_experiments_per_variant': 30,  # 每个变体运行30次（提高统计可靠性）
    'parallel_jobs': 4,  # 并行任务数
    'timeout_minutes': 60,  # 单次实验超时时间
    'save_intermediate_results': True,
    'log_level': 'INFO',
    'random_seeds': list(range(42, 72)),  # 30个不同的随机种子
    'statistical_power': 0.8,  # 统计功效
    'effect_size_threshold': 0.3  # 最小可检测效应量
}

# 评估指标配置
EVALUATION_METRICS = {
    'primary_metrics': [
        'annual_return',
        'sharpe_ratio', 
        'max_drawdown',
        'win_rate'
    ],
    'secondary_metrics': [
        'information_ratio',
        'calmar_ratio',
        'sortino_ratio',
        'volatility',
        'var_95',
        'cvar_95'
    ],
    'algorithm_specific_metrics': [
        'search_convergence_speed',
        'expression_complexity',
        'search_tree_depth',
        'exploration_coverage'
    ]
}

# 统计检验配置
STATISTICAL_CONFIG = {
    'significance_level': 0.05,
    'confidence_level': 0.95,
    'bootstrap_samples': 1000,
    'multiple_comparison_correction': 'bonferroni'
}

# CSV输出格式配置
CSV_OUTPUT_CONFIG = {
    'decimal_places': 6,
    'date_format': '%Y-%m-%d',
    'encoding': 'utf-8',
    'index': True,
    'float_format': '%.6f'
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'Set2',
    'save_formats': ['png', 'pdf']
}

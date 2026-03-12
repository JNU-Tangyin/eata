"""
EATA消融实验配置文件
定义6个变体的具体配置参数
"""

# 基准配置
BASE_CONFIG = {
    'name': 'EATA-Full',
    'description': '完整的EATA模型，作为基准',
    'modifications': {},
    'expected_performance': 'baseline'
}

# 变体配置
ABLATION_CONFIGS = {
    'EATA-NoNN': {
        'name': 'EATA-NoNN',
        'description': '无神经网络引导 - 移除神经网络先验，纯MCTS搜索',
        'modifications': {
            'alpha': 0.0,  # 在mcts.py中强制设置alpha=0
            'target_file': 'eata_agent/mcts.py',
            'target_line': 264,
            'modification_type': 'parameter_override'
        },
        'hypothesis': '搜索效率大幅下降，无法在有限步数内找到复杂且有效的公式',
        'expected_performance': {
            'annual_return': '-20% to -30%',
            'search_convergence': '-60%',
            'expression_quality': 'significantly lower'
        }
    },
    
    'EATA-NoMem': {
        'name': 'EATA-NoMem',
        'description': '无进化记忆 - 移除历史知识传承机制',
        'modifications': {
            'num_transplant': 0,  # 在model.py中设置
            'num_aug': 0,         # 在model.py中设置
            'target_file': 'eata_agent/model.py',
            'target_lines': [20, 23],
            'modification_type': 'parameter_override'
        },
        'hypothesis': '无法利用历史知识，每个窗口从零开始，模型无法捕捉市场长期特征',
        'expected_performance': {
            'long_term_sharpe': '-0.4 to -0.6',
            'adaptation_ability': '-50%',
            'knowledge_retention': 'none'
        }
    },
    
    'EATA-Simple': {
        'name': 'EATA-Simple',
        'description': '简单奖励 - 替换复杂的Wasserstein距离为简单收益差',
        'modifications': {
            'reward_function': 'simple_mae',  # 在agent.py中替换wasserstein_distance
            'target_file': 'agent.py',
            'target_line': 167,
            'modification_type': 'function_replacement'
        },
        'hypothesis': '对分布的鲁棒性变差，容易受到极端行情噪声点影响',
        'expected_performance': {
            'distribution_robustness': '-60%',
            'extreme_event_handling': 'poor',
            'noise_sensitivity': 'high'
        }
    },
    
    'EATA-LowExplore': {
        'name': 'EATA-LowExplore',
        'description': '低探索机制 - 显著降低UCT探索强度，测试探索-利用平衡',
        'modifications': {
            'exploration_rate': 0.01,  # 在mcts.py中设置exploration_rate=0.01
            'target_file': 'eata_agent/mcts.py',
            'target_line': 32,
            'modification_type': 'parameter_override'
        },
        'hypothesis': '显著降低UCT探索强度，测试探索-利用平衡对搜索质量的影响，预期局部最优陷阱增加',
        'expected_performance': {
            'global_optimum_discovery': '-40% to -60%',
            'expression_diversity': '-30% to -50%',
            'local_optimum_trap_rate': '+30% to +50%'
        }
    },
    
    'EATA-NoMCTS': {
        'name': 'EATA-NoMCTS',
        'description': '无蒙特卡洛模拟 - 纯神经网络引导，移除随机模拟',
        'modifications': {
            'alpha': 1.0,  # 在mcts.py中强制设置alpha=1.0
            'target_file': 'eata_agent/mcts.py',
            'target_line': 264,
            'modification_type': 'parameter_override'
        },
        'hypothesis': '完全依赖神经网络，移除MCTS随机模拟，搜索变得过于确定性，缺乏探索多样性',
        'expected_performance': {
            'search_determinism': 'high',
            'exploration_diversity': '-70%',
            'local_optimum_risk': '+80%'
        }
    },
    
    'EATA-NoExplore': {
        'name': 'EATA-NoExplore',
        'description': '无探索机制 - 移除MCTS探索，纯利用已知最优解',
        'modifications': {
            'exploration_rate': 0.0,  # 在model.py中强制设置exploration_rate=0.0
            'target_file': 'eata_agent/model.py',
            'target_method': 'run',
            'modification_type': 'parameter_override'
        },
        'hypothesis': '缺乏探索能力，容易陷入局部最优，无法发现更好的交易策略',
        'expected_performance': {
            'exploration_diversity': '-90%',
            'local_optimum_risk': '+95%',
            'strategy_innovation': 'minimal'
        }
    }
}

# 实验设置
EXPERIMENT_SETTINGS = {
    'data_split': {
        'train_start': '2010-01-01',
        'train_end': '2018-12-31',
        'validation_start': '2019-01-01',
        'validation_end': '2020-12-31',
        'test_start': '2021-01-01',
        'test_end': '2022-12-31'
    },
    'evaluation_metrics': [
        'annual_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'information_ratio',
        'calmar_ratio',
        'sortino_ratio',
        'volatility'
    ],
    'statistical_tests': [
        'wilcoxon_signed_rank',
        'paired_t_test',
        'bootstrap_confidence_interval'
    ],
    'num_runs': 50,  # 每个变体运行50次取平均
    'confidence_level': 0.95,
    'random_seed': 42
}

# CSV输出配置
CSV_CONFIG = {
    'performance_summary': 'performance_summary.csv',
    'detailed_metrics': 'detailed_metrics.csv',
    'statistical_tests': 'statistical_tests.csv',
    'variant_comparison': 'variant_comparison.csv',
    'time_series_results': 'time_series_results.csv'
}

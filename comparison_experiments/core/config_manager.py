"""
实验配置管理器

管理所有实验的配置参数，生成笛卡尔积组合。
"""

from typing import Dict, List, Any, Iterator
import itertools


class ConfigManager:
    """实验配置管理器"""
    
    def __init__(self):
        """初始化配置管理器"""
        self.algorithms = {}
        self.hyperparams = {}
        self.datasets = {}
        
    def register_algorithm(self, name: str, algorithm_class, default_config: Dict[str, Any] = None):
        """
        注册算法
        
        Args:
            name: 算法名称
            algorithm_class: 算法类
            default_config: 默认配置
        """
        self.algorithms[name] = {
            'class': algorithm_class,
            'default_config': default_config or {}
        }
    
    def set_hyperparams(self, hyperparams: Dict[str, List[Any]]):
        """
        设置超参数搜索空间
        
        Args:
            hyperparams: 超参数字典，值为列表表示搜索空间
        """
        self.hyperparams = hyperparams
    
    def set_datasets(self, datasets: List[str]):
        """
        设置数据集列表
        
        Args:
            datasets: 数据集名称列表
        """
        self.datasets = {name: name for name in datasets}
    
    def get_experiment_configs(self) -> Iterator[Dict[str, Any]]:
        """
        生成所有实验配置的笛卡尔积
        
        Yields:
            实验配置字典
        """
        # 生成超参数组合
        hyperparam_names = list(self.hyperparams.keys())
        hyperparam_values = list(self.hyperparams.values())
        
        for algorithm_name, algorithm_info in self.algorithms.items():
            for dataset_name in self.datasets.keys():
                for hyperparam_combo in itertools.product(*hyperparam_values):
                    # 构建配置
                    config = {
                        'algorithm': algorithm_name,
                        'algorithm_class': algorithm_info['class'],
                        'dataset': dataset_name,
                        'stock': dataset_name,  # 兼容性
                    }
                    
                    # 添加默认配置
                    config.update(algorithm_info['default_config'])
                    
                    # 添加当前超参数组合
                    for param_name, param_value in zip(hyperparam_names, hyperparam_combo):
                        config[param_name] = param_value
                    
                    yield config
    
    def count_experiments(self) -> int:
        """计算实验总数"""
        num_algorithms = len(self.algorithms)
        num_datasets = len(self.datasets)
        
        # 计算超参数组合数
        num_hyperparams = 1
        for param_values in self.hyperparams.values():
            num_hyperparams *= len(param_values)
        
        return num_algorithms * num_datasets * num_hyperparams
    
    def get_quick_configs(self, max_experiments: int = 12) -> List[Dict[str, Any]]:
        """
        获取快速验证用的配置子集
        
        Args:
            max_experiments: 最大实验数量
            
        Returns:
            配置列表
        """
        all_configs = list(self.get_experiment_configs())
        
        if len(all_configs) <= max_experiments:
            return all_configs
        
        # 均匀采样
        step = len(all_configs) // max_experiments
        return [all_configs[i] for i in range(0, len(all_configs), step)][:max_experiments]
    
    def print_summary(self):
        """打印配置摘要"""
        print("🔧 实验配置摘要")
        print("=" * 50)
        
        print(f"算法数量: {len(self.algorithms)}")
        for name in self.algorithms.keys():
            print(f"  - {name}")
        
        print(f"\n数据集数量: {len(self.datasets)}")
        for name in self.datasets.keys():
            print(f"  - {name}")
        
        print(f"\n超参数配置:")
        for param_name, param_values in self.hyperparams.items():
            print(f"  - {param_name}: {param_values}")
        
        total_experiments = self.count_experiments()
        print(f"\n总实验数量: {total_experiments}")
        print("=" * 50)


def create_default_config() -> ConfigManager:
    """创建默认的实验配置 - 包含所有算法"""
    from algorithms.eata_algorithm import EATAAlgorithm
    from algorithms.transformer_algorithm import TransformerAlgorithm
    from algorithms.ce_dnn_algorithm import CEDNNAlgorithm
    from algorithms.stocknet_algorithm import StockNetAlgorithm
    from algorithms.lstm_algorithm import LSTMAlgorithm
    from algorithms.espmp_algorithm import ESMPAlgorithm
    from algorithms.scl_dnn_algorithm import SCLDNNAlgorithm
    from algorithms.dual_dnn_algorithm import DualDNNAlgorithm
    
    config_manager = ConfigManager()
    
    # 注册所有算法 - 统一训练次数1000次
    # 1. 原有算法
    config_manager.register_algorithm('EATA', EATAAlgorithm, {'windows': 1000})
    config_manager.register_algorithm('CE-DNN', CEDNNAlgorithm, {'max_iter': 1000})
    
    # 2. 传统模态融合方法（TMF）
    config_manager.register_algorithm('StockNet', StockNetAlgorithm, {'epochs': 1000})
    config_manager.register_algorithm('LSTM', LSTMAlgorithm, {'epochs': 1000})
    config_manager.register_algorithm('Transformer', TransformerAlgorithm, {'epochs': 1000})
    
    # 3. 先进神经网络方法（SoTA）
    config_manager.register_algorithm('ESPMP', ESMPAlgorithm, {'epochs': 1000})
    config_manager.register_algorithm('SCL-DNN', SCLDNNAlgorithm, {'epochs': 1000})
    config_manager.register_algorithm('DUAL-DNN', DualDNNAlgorithm, {'epochs': 1000})
    
    # 设置1种超参数组合 (1×1×1×1 = 1)
    config_manager.set_hyperparams({
        'lookback': [50],                        # 1种回看窗口
        'lookahead': [10],                       # 1种前瞻窗口  
        'stride': [1],                           # 1种步长
        'depth': [300]                           # 1种深度
    })
    
    # 设置多个数据集进行全面测试 (每个都有3774条记录)
    config_manager.set_datasets(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'INTC'])
    
    return config_manager


def create_quick_config() -> ConfigManager:
    """创建快速验证配置 - 包含所有算法"""
    from algorithms.eata_algorithm import EATAAlgorithm
    from algorithms.transformer_algorithm import TransformerAlgorithm
    from algorithms.ce_dnn_algorithm import CEDNNAlgorithm
    from algorithms.stocknet_algorithm import StockNetAlgorithm
    from algorithms.lstm_algorithm import LSTMAlgorithm
    from algorithms.espmp_algorithm import ESMPAlgorithm
    from algorithms.scl_dnn_algorithm import SCLDNNAlgorithm
    from algorithms.dual_dnn_algorithm import DualDNNAlgorithm
    
    config_manager = ConfigManager()
    
    # 注册所有算法（快速版本 - 统一训练次数50次）
    # 1. 原有算法
    config_manager.register_algorithm('EATA', EATAAlgorithm, {'windows': 50})
    config_manager.register_algorithm('CE-DNN', CEDNNAlgorithm, {'max_iter': 50})
    
    # 2. 传统模态融合方法（TMF）
    config_manager.register_algorithm('StockNet', StockNetAlgorithm, {'epochs': 50})
    config_manager.register_algorithm('LSTM', LSTMAlgorithm, {'epochs': 50})
    config_manager.register_algorithm('Transformer', TransformerAlgorithm, {'epochs': 50})
    
    # 3. 先进神经网络方法（SoTA）
    config_manager.register_algorithm('ESPMP', ESMPAlgorithm, {'epochs': 50})
    config_manager.register_algorithm('SCL-DNN', SCLDNNAlgorithm, {'epochs': 50})
    config_manager.register_algorithm('DUAL-DNN', DualDNNAlgorithm, {'epochs': 50})
    
    # 设置超参数（单一组合）
    config_manager.set_hyperparams({
        'lookback': [50],
        'lookahead': [10],
        'stride': [1],
        'depth': [300]
    })
    
    # 设置数据集
    config_manager.set_datasets(['AAPL'])
    
    return config_manager

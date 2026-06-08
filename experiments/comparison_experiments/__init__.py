import warnings

# 忽略 urllib3 在 LibreSSL 环境下关于 NotOpenSSL 的兼容性提示
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# 禁用PyTorch MPS pin_memory警告
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but not supported on MPS now")
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*", category=UserWarning)

# 禁用statsmodels相关警告
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# 禁用sklearn FutureWarning
warnings.filterwarnings("ignore", message="`BaseEstimator._validate_data` is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# 注释掉缺失的core模块导入
# from .core.base_algorithm import BaseAlgorithm
# from .core.experiment_runner import ExperimentRunner
# from .core.config_manager import ConfigManager

# __all__ = [
#     'BaseAlgorithm',
#     'ExperimentRunner', 
#     'ConfigManager'
# ]

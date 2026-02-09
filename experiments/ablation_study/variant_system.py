"""
EATA变体系统架构 - 统一的变体参数管理和执行框架
设计目标：
1. 统一的变体参数管理
2. 清晰的变体接口定义
3. 可靠的参数传递机制
4. 完整的测试验证体系
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass
import copy

@dataclass
class VariantConfig:
    """
    变体配置类 - 统一管理所有变体参数
    """
    # 基础信息
    name: str
    description: str
    
    # 核心参数
    profit_loss_weight: Optional[float] = None
    exploration_rate: Optional[float] = None
    num_transplant: Optional[int] = None
    distance_function: Optional[str] = None
    
    # 训练参数
    train_size: Optional[int] = None
    learning_rate: Optional[float] = None
    
    # 其他参数
    extra_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，过滤None值"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None and key not in ['name', 'description', 'extra_params']:
                result[key] = value
        
        # 添加额外参数
        if self.extra_params:
            result.update(self.extra_params)
            
        return result
    
    def get_debug_info(self) -> str:
        """获取调试信息"""
        params = self.to_dict()
        if params:
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            return f"{self.name}: {param_str}"
        else:
            return f"{self.name}: 无特殊参数（使用默认值）"

class BaseVariant(ABC):
    """
    变体基类 - 定义统一的变体接口
    """
    
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        初始化变体
        
        Args:
            df: 股票数据DataFrame
            **kwargs: 其他参数
        """
        self.df = df
        self.kwargs = kwargs
        self.config = self._create_config()
        
        print(f"🔧 [变体系统] 初始化 {self.config.name}")
        print(f"   配置: {self.config.get_debug_info()}")
    
    @abstractmethod
    def _create_config(self) -> VariantConfig:
        """
        创建变体配置 - 子类必须实现
        
        Returns:
            VariantConfig: 变体配置对象
        """
        pass
    
    def run_backtest(self, train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        运行回测 - 统一的执行流程
        
        Args:
            train_df: 训练数据
            test_df: 测试数据
            ticker: 股票代码
            
        Returns:
            dict: 回测结果
        """
        print(f"🚀 [变体系统] 开始 {self.config.name} 回测 - {ticker}")
        print(f"   参数配置: {self.config.get_debug_info()}")
        
        try:
            # 合并数据
            combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
            
            # 执行回测
            result = self._execute_backtest(combined_df, ticker)
            
            print(f"✅ [变体系统] {self.config.name} 回测完成")
            return result
            
        except Exception as e:
            print(f"❌ [变体系统] {self.config.name} 回测失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _execute_backtest(self, combined_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        执行回测的具体实现
        
        Args:
            combined_df: 合并后的数据
            ticker: 股票代码
            
        Returns:
            dict: 回测结果
        """
        # 导入核心回测函数
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(__file__))
        sys.path.insert(0, project_root)
        
        from predict import run_eata_core_backtest
        
        # 准备变体参数
        variant_params = self.config.to_dict()
        
        print(f"🔧 [变体系统] 传递参数: {variant_params}")
        
        # 执行核心回测
        core_metrics, portfolio_df = run_eata_core_backtest(
            stock_df=combined_df,
            ticker=ticker,
            lookback=50,
            lookahead=10,
            stride=1,
            depth=300,
            variant_params=variant_params,  # 统一使用variant_params传递
            pre_configured_agent=None
        )
        
        return {
            'variant': self.config.name,
            'ticker': ticker,
            'annual_return': core_metrics.get('Annual Return (AR)', 0.0),
            'sharpe_ratio': core_metrics.get('Sharpe Ratio', 0.0),
            'max_drawdown': core_metrics.get('Max Drawdown (MDD)', 0.0),
            'win_rate': core_metrics.get('Win Rate', 0.0),
            'volatility': core_metrics.get('Volatility (Annual)', 0.0),
            'rl_reward': core_metrics.get('Average RL Reward', 0.0),
            'portfolio_df': portfolio_df,
            'variant_config': self.config
        }

class VariantParameterApplier:
    """
    变体参数应用器 - 统一的参数设置机制
    """
    
    @staticmethod
    def apply_to_agent(agent, variant_params: Dict[str, Any]) -> bool:
        """
        将变体参数应用到Agent实例
        
        Args:
            agent: Agent实例
            variant_params: 变体参数字典
            
        Returns:
            bool: 是否应用成功
        """
        if not variant_params:
            print("🔧 [参数应用器] 无变体参数需要应用")
            return True
            
        print(f"🔧 [参数应用器] 开始应用变体参数: {variant_params}")
        
        success_count = 0
        total_count = len(variant_params)
        
        for param_name, param_value in variant_params.items():
            try:
                success = VariantParameterApplier._apply_single_parameter(
                    agent, param_name, param_value
                )
                if success:
                    success_count += 1
                    print(f"   ✅ {param_name} = {param_value}")
                else:
                    print(f"   ❌ {param_name} = {param_value} (应用失败)")
                    
            except Exception as e:
                print(f"   ❌ {param_name} = {param_value} (异常: {e})")
        
        success_rate = success_count / total_count if total_count > 0 else 1.0
        print(f"🔧 [参数应用器] 应用完成: {success_count}/{total_count} ({success_rate:.1%})")
        
        return success_rate >= 0.8  # 80%以上成功率认为应用成功
    
    @staticmethod
    def _apply_single_parameter(agent, param_name: str, param_value: Any) -> bool:
        """
        应用单个参数
        
        Args:
            agent: Agent实例
            param_name: 参数名
            param_value: 参数值
            
        Returns:
            bool: 是否应用成功
        """
        try:
            # 1. 尝试在hyperparams上设置
            if hasattr(agent, 'hyperparams'):
                setattr(agent.hyperparams, param_name, param_value)
            
            # 2. 尝试在engine.args上设置
            if hasattr(agent, 'engine') and hasattr(agent.engine, 'args'):
                setattr(agent.engine.args, param_name, param_value)
            
            # 3. 设置变体标识（用于Engine检测）
            if hasattr(agent, 'engine') and hasattr(agent.engine, 'model'):
                setattr(agent.engine.model, f'_variant_{param_name}', param_value)
            
            # 4. 在agent上设置变体标识
            setattr(agent, f'_variant_{param_name}', param_value)
            
            # 5. 特殊处理distance_function参数
            if param_name == 'distance_function':
                VariantParameterApplier._apply_distance_function(agent, param_value)
            
            # 6. 特殊处理num_transplant参数
            if param_name == 'num_transplant':
                VariantParameterApplier._apply_num_transplant(agent, param_value)
            
            # 7. 特殊处理exploration_rate参数
            if param_name == 'exploration_rate':
                VariantParameterApplier._apply_exploration_rate(agent, param_value)
            
            return True
            
        except Exception as e:
            print(f"     参数设置异常: {e}")
            return False
    
    @staticmethod
    def _apply_distance_function(agent, distance_function: str):
        """
        特殊处理distance_function参数
        确保参数能传递到实际使用的地方
        """
        try:
            # 在agent上设置，供agent.py中的检查逻辑使用
            agent._variant_distance_function = distance_function
            
            # 如果有model，也在model上设置
            if hasattr(agent, 'engine') and hasattr(agent.engine, 'model'):
                agent.engine.model._variant_distance_function = distance_function
            
            # 在hyperparams中设置，供其他组件使用
            if hasattr(agent, 'hyperparams'):
                agent.hyperparams.distance_function = distance_function
            
            print(f"     特殊处理distance_function: {distance_function}")
            
        except Exception as e:
            print(f"     distance_function特殊处理失败: {e}")
    
    @staticmethod
    def _apply_num_transplant(agent, num_transplant: int):
        """
        特殊处理num_transplant参数
        确保参数能传递到MCTS调用中
        """
        try:
            # 在agent上设置，供MCTS使用
            agent._variant_num_transplant = num_transplant
            
            # 如果有model，也在model上设置
            if hasattr(agent, 'engine') and hasattr(agent.engine, 'model'):
                agent.engine.model._variant_num_transplant = num_transplant
            
            # 在hyperparams中设置
            if hasattr(agent, 'hyperparams'):
                agent.hyperparams.num_transplant = num_transplant
            
            # 尝试在engine中设置，供MCTS调用时使用
            if hasattr(agent, 'engine'):
                agent.engine._variant_num_transplant = num_transplant
            
            print(f"     特殊处理num_transplant: {num_transplant}")
            
        except Exception as e:
            print(f"     num_transplant特殊处理失败: {e}")
    
    @staticmethod
    def _apply_exploration_rate(agent, exploration_rate: float):
        """
        特殊处理exploration_rate参数
        确保参数能传递到Model.simulate中
        """
        try:
            # 在agent上设置，供Model使用
            agent._variant_exploration_rate = exploration_rate
            
            # 如果有model，也在model上设置
            if hasattr(agent, 'engine') and hasattr(agent.engine, 'model'):
                agent.engine.model._variant_exploration_rate = exploration_rate
            
            # 在hyperparams中设置
            if hasattr(agent, 'hyperparams'):
                agent.hyperparams.exploration_rate = exploration_rate
            
            # 尝试在engine中设置，供Model.simulate使用
            if hasattr(agent, 'engine'):
                agent.engine._variant_exploration_rate = exploration_rate
            
            print(f"     特殊处理exploration_rate: {exploration_rate}")
            
        except Exception as e:
            print(f"     exploration_rate特殊处理失败: {e}")

class VariantTester:
    """
    变体测试器 - 统一的测试验证体系
    """
    
    @staticmethod
    def test_variant(variant_class, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        测试变体实现
        
        Args:
            variant_class: 变体类
            test_data: 测试数据
            
        Returns:
            dict: 测试结果
        """
        print(f"🧪 [变体测试器] 开始测试 {variant_class.__name__}")
        
        try:
            # 创建变体实例
            variant = variant_class(df=test_data)
            
            # 运行回测
            result = variant.run_backtest(test_data, test_data, "TEST")
            
            # 验证结果
            is_valid = VariantTester._validate_result(result)
            
            return {
                'success': True,
                'variant_name': variant.config.name,
                'config': variant.config,
                'result_valid': is_valid,
                'result': result
            }
            
        except Exception as e:
            print(f"❌ [变体测试器] 测试失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'variant_name': getattr(variant_class, '__name__', 'Unknown')
            }
    
    @staticmethod
    def _validate_result(result: Dict[str, Any]) -> bool:
        """
        验证回测结果的有效性
        
        Args:
            result: 回测结果
            
        Returns:
            bool: 结果是否有效
        """
        required_keys = ['annual_return', 'sharpe_ratio', 'variant_config']
        return all(key in result for key in required_keys)

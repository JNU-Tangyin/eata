"""
EATA-NoExplore: 无探索机制变体
移除UCT探索项，设置exploration_rate=0，纯贪心搜索
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.agent import Agent
import pandas as pd
import numpy as np

class EATANoExplore:
    """
    低探索机制的EATA变体
    通过设置exploration_rate=0.01显著降低UCT探索强度，验证探索机制的价值
    """
    
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        初始化低探索机制的EATA模型
        """
        self.name = "EATA-NoExplore"
        self.description = "无探索机制 - 完全禁用UCT探索强度，纯贪心搜索"
        
        # 创建Agent实例
        self.agent = Agent(
            df=df,
            lookback=kwargs.get('lookback', 100),
            lookahead=kwargs.get('lookahead', 20),
            stride=kwargs.get('stride', 1),
            depth=kwargs.get('depth', 300)
        )
        
        # 应用消融修改
        self._apply_modifications()
        
        self.modifications = {
            'exploration_rate': 0.0,
            'target_file': 'eata_agent/mcts.py',
            'target_line': 32,
            'modification_type': 'parameter_override'
        }
        
    def _apply_modifications(self):
        """
        应用消融修改：设置exploration_rate=0.0完全禁用UCT探索强度
        """
        try:
            # 修改Agent内部的超参数
            if hasattr(self.agent, 'hyperparams'):
                # 设置exploration_rate为0.0，完全禁用探索
                self.agent.hyperparams.exploration_rate = 0.0
                print(f"{self.name}: 已设置hyperparams.exploration_rate=0.0")
            
            # 关键：设置engine.model的_variant_exploration_rate属性
            if hasattr(self.agent, 'engine') and self.agent.engine:
                if hasattr(self.agent.engine, 'model'):
                    self.agent.engine.model._variant_exploration_rate = 0.0
                    print(f"{self.name}: 已设置engine.model._variant_exploration_rate=0.0")
            
            # 为NoExplore变体添加特定标识，确保不同的计算路径
            self.agent._variant_id = "NoExplore"
            print(f"{self.name}: 已设置变体标识为NoExplore")
            
        except Exception as e:
            print(f"{self.name}: 应用修改时出错: {e}")
    
    def run_backtest(self, train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
        """
        运行回测 - 使用与对比实验相同的核心回测逻辑
        """
        try:
            print(f"运行{self.name}回测 - {ticker}")
            print(f"   修改: exploration_rate=0.0 (无探索机制)")
            print(f"🔧 [调试] 开始应用{self.name}的修改...")
            
            # 导入并使用与对比实验相同的核心回测函数
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.insert(0, project_root)
            
            from predict import run_eata_core_backtest
            
            # 合并训练和测试数据
            combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
            
            # 使用新架构的统一参数传递机制
            variant_params = {'exploration_rate': 0.0}
            print(f"🔧 [EATA-NoExplore] 使用新架构传递参数: {variant_params}")
            
            # 使用核心回测函数，通过新架构传递exploration_rate=0.0
            core_metrics, portfolio_df = run_eata_core_backtest(
                stock_df=combined_df,
                ticker=ticker,
                lookback=50,
                lookahead=10,
                stride=1,
                depth=300,
                variant_params=variant_params,  # 使用新架构传递参数
                pre_configured_agent=None  # 让新架构处理参数应用
            )
            
            # 转换指标格式
            annual_return = core_metrics.get('Annual Return (AR)', 0.0)
            sharpe_ratio = core_metrics.get('Sharpe Ratio', 0.0)
            max_drawdown = core_metrics.get('Max Drawdown (MDD)', 0.0)
            win_rate = core_metrics.get('Win Rate', 0.0)
            volatility = core_metrics.get('Volatility (Annual)', 0.0)
            avg_rl_reward = core_metrics.get('Average RL Reward', 0.0)
            
            return {
                'variant': self.name,
                'ticker': ticker,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': volatility,
                'rl_reward': avg_rl_reward,
                'modifications': self.modifications
            }
            
            print(f"{self.name}回测完成 - 年化收益: {annual_return:.4f}")
            
        except Exception as e:
            print(f"{self.name}回测失败: {str(e)}")
            return {
                'variant': self.name,
                'ticker': ticker,
                'error': str(e),
                'modifications': self.modifications
            }
    
    def _calculate_returns(self, test_df: pd.DataFrame, trading_signal: int):
        """计算收益率序列"""
        if len(test_df) < 2:
            return np.array([0.0])
            
        prices = test_df['close'].values
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = price_returns * trading_signal
        
        return strategy_returns
    
    def _calculate_metrics(self, returns: np.ndarray):
        """计算性能指标"""
        if len(returns) == 0:
            return {
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0
            }
        
        annual_return = np.mean(returns) * 252
        sharpe_ratio = annual_return / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0
        volatility = np.std(returns) * np.sqrt(252)
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility
        }
    
    def get_variant_info(self):
        """获取变体信息"""
        return {
            'name': self.name,
            'description': self.description,
            'modifications': self.modifications,
            'hypothesis': '显著降低UCT探索强度，测试探索-利用平衡对搜索质量的影响，预期局部最优陷阱增加',
            'expected_performance': {
                'global_optimum_discovery': '-40% to -60%',
                'expression_diversity': '-30% to -50%',
                'local_optimum_trap_rate': '+30% to +50%'
            }
        }

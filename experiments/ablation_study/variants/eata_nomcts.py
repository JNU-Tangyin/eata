"""
EATA-NoMCTS: 无蒙特卡洛模拟变体
纯神经网络引导，设置alpha=1.0，移除随机模拟
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pandas as pd
import numpy as np

class EATANoMCTS:
    """
    无蒙特卡洛模拟的EATA变体
    通过设置alpha=1.0完全依赖神经网络，移除MCTS随机模拟
    """
    
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        初始化无蒙特卡洛模拟的EATA模型
        """
        self.name = "EATA-NoMCTS"
        self.description = "无蒙特卡洛模拟 - 纯神经网络引导，移除随机模拟"
        
        self.modifications = {
            'alpha': 1.0,
            'target_file': 'eata_agent/mcts.py',
            'target_line': 264,
            'modification_type': 'parameter_override'
        }
        
    def run_backtest(self, train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
        """
        运行回测 - 使用与对比实验相同的核心回测逻辑
        """
        try:
            print(f"运行{self.name}回测 - {ticker}")
            print(f"   修改: alpha=1.0 (无蒙特卡洛模拟，纯神经网络)")
            
            # 导入并使用与对比实验相同的核心回测函数
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.insert(0, project_root)
            
            from predict import run_eata_core_backtest
            
            # 合并训练和测试数据
            combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
            
            # 使用核心回测函数，传入变体参数（真正移除MCTS，纯神经网络）
            variant_params = {
                'skip_mcts': True,  # 🔧 新增：完全跳过MCTS搜索
                'use_nn_direct': True  # 🔧 新增：直接用神经网络生成表达式
            }
            core_metrics, portfolio_df = run_eata_core_backtest(
                stock_df=combined_df,
                ticker=ticker,
                lookback=50,
                lookahead=10,
                stride=1,
                depth=300,
                variant_params=variant_params,  # 只传入需要修改的参数
                pre_configured_agent=None  # 让主程序自己创建Agent
            )
            
            # 转换指标格式
            annual_return = core_metrics.get('Annual Return (AR)', 0.0)
            sharpe_ratio = core_metrics.get('Sharpe Ratio', 0.0)
            max_drawdown = core_metrics.get('Max Drawdown (MDD)', 0.0)
            win_rate = core_metrics.get('Win Rate', 0.0)
            volatility = core_metrics.get('Volatility (Annual)', 0.0)
            avg_rl_reward = core_metrics.get('Average RL Reward', 0.0)
            
            print(f"{self.name}回测完成 - 年化收益: {annual_return:.4f}")
            
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
            'hypothesis': '完全依赖神经网络，移除MCTS随机模拟，搜索变得过于确定性，缺乏探索多样性',
            'expected_performance': {
                'search_determinism': 'high',
                'exploration_diversity': '-70%',
                'local_optimum_risk': '+80%'
            }
        }

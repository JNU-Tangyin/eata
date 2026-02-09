"""
EATA-DistanceCompare: 距离度量对比变体
对比多种距离度量方法，验证Wasserstein距离的优势
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.agent import Agent
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

class EATADistanceCompare:
    """
    距离度量对比的EATA变体
    通过对比多种距离度量方法验证Wasserstein距离的独特价值
    """
    
    def __init__(self, df: pd.DataFrame, distance_method: str = 'l2', **kwargs):
        """
        初始化距离度量对比的EATA模型
        
        Args:
            distance_method: 距离度量方法 ('l1', 'l2', 'kl', 'cosine')
        """
        self.distance_method = distance_method
        self.name = f"EATA-{distance_method.upper()}Distance"
        self.description = f"使用{distance_method.upper()}距离替代Wasserstein距离的奖励机制"
        
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
            'reward_function': f'{distance_method}_distance',
            'target_file': 'agent.py',
            'target_line': 167,
            'modification_type': 'function_replacement'
        }
        
    def _apply_modifications(self):
        """
        应用消融修改：替换距离度量函数
        """
        try:
            # 重写Agent的奖励计算方法
            original_method = self.agent._calculate_rl_reward_and_signal
            
            def alternative_distance_calculation(prediction_distribution, lookahead_ground_truth):
                """
                使用不同距离度量的奖励计算
                """
                try:
                    # 计算决策信号（保持原逻辑）
                    q_low = np.percentile(prediction_distribution, 25)
                    q_high = np.percentile(prediction_distribution, 75)
                    
                    intended_signal = 0
                    if q_low > 0:
                        intended_signal = 1
                        print(f"  [决策] 预测分布的 25% 分位数 > 0，生成意图信号: 买入")
                    elif q_high < 0:
                        intended_signal = -1
                        print(f"  [决策] 预测分布的 75% 分位数 < 0，生成意图信号: 卖出")
                    else:
                        print("  [决策] 预测分布跨越零点，信号不明确，生成意图信号: 持有")
                    
                    # 使用不同的距离度量计算奖励
                    actual_returns = lookahead_ground_truth.T[3, :]
                    distance = self._calculate_distance(prediction_distribution, actual_returns)
                    rl_reward = 1 / (1 + distance)
                    
                    print(f"  [{self.distance_method.upper()}距离] Distance: {distance:.6f}, 奖励: {rl_reward:.6f}")
                    
                    return rl_reward, intended_signal
                    
                except Exception as e:
                    print(f"--- 🚨 {self.distance_method.upper()}距离计算中捕获到错误 🚨 ---")
                    print(f"错误信息: {e}")
                    return 0.0, 0
            
            # 替换原方法
            self.agent._calculate_rl_reward_and_signal = alternative_distance_calculation
            print(f"🔧 {self.name}: 已替换为{self.distance_method.upper()}距离度量")
            
        except Exception as e:
            print(f"⚠️ {self.name}: 应用修改时出错: {e}")
    
    def _calculate_distance(self, pred_dist: np.ndarray, actual_dist: np.ndarray) -> float:
        """
        根据指定方法计算距离
        """
        # 确保两个分布长度相同
        min_len = min(len(pred_dist), len(actual_dist))
        pred = pred_dist[:min_len]
        actual = actual_dist[:min_len]
        
        if self.distance_method == 'l1':
            # L1距离 (曼哈顿距离)
            return np.mean(np.abs(pred - actual))
            
        elif self.distance_method == 'l2':
            # L2距离 (欧几里得距离)
            return np.sqrt(np.mean((pred - actual) ** 2))
            
        elif self.distance_method == 'kl':
            # KL散度 (需要处理概率分布)
            try:
                # 转换为概率分布
                pred_prob = np.abs(pred) / (np.sum(np.abs(pred)) + 1e-10)
                actual_prob = np.abs(actual) / (np.sum(np.abs(actual)) + 1e-10)
                
                # 添加小的平滑项避免0概率
                pred_prob += 1e-10
                actual_prob += 1e-10
                
                return entropy(pred_prob, actual_prob)
            except:
                # 如果KL散度计算失败，回退到L2距离
                return np.sqrt(np.mean((pred - actual) ** 2))
                
        elif self.distance_method == 'cosine':
            # 余弦距离
            try:
                dot_product = np.dot(pred, actual)
                norm_pred = np.linalg.norm(pred)
                norm_actual = np.linalg.norm(actual)
                
                if norm_pred == 0 or norm_actual == 0:
                    return 1.0  # 最大余弦距离
                
                cosine_sim = dot_product / (norm_pred * norm_actual)
                return 1 - cosine_sim  # 转换为距离
            except:
                return 1.0
                
        else:
            # 默认使用L2距离
            return np.sqrt(np.mean((pred - actual) ** 2))
    
    def run_backtest(self, train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
        """
        运行回测
        """
        try:
            print(f"🚀 运行{self.name}回测 - {ticker}")
            print(f"   修改: 使用{self.distance_method.upper()}距离替代Wasserstein距离")
            
            # 使用修改后的Agent进行回测
            trading_signal, rl_reward = self.agent.criteria(test_df, shares_held=0)
            
            # 计算指标
            returns = self._calculate_returns(test_df, trading_signal)
            metrics = self._calculate_metrics(returns)
            
            results = {
                'variant': self.name,
                'ticker': ticker,
                'trading_signals': trading_signal,
                'returns': returns,
                'metrics': metrics,
                'rl_reward': rl_reward,
                'modifications': self.modifications
            }
            
            print(f"✅ {self.name}回测完成 - 年化收益: {annual_return:.4f}")
            
        except Exception as e:
            print(f"❌ {self.name}回测失败: {str(e)}")
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
            'hypothesis': f'使用{self.distance_method.upper()}距离可能无法像Wasserstein距离那样有效捕捉分布差异，特别是在极端市场事件中',
            'expected_performance': {
                'distribution_sensitivity': 'lower than Wasserstein',
                'extreme_event_handling': 'potentially worse',
                'overall_performance': 'depends on distance method'
            }
        }


# 创建不同距离度量的变体类
class EATAL1Distance(EATADistanceCompare):
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df, distance_method='l1', **kwargs)

class EATAL2Distance(EATADistanceCompare):
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df, distance_method='l2', **kwargs)

class EATAKLDistance(EATADistanceCompare):
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df, distance_method='kl', **kwargs)

class EATACosineDistance(EATADistanceCompare):
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df, distance_method='cosine', **kwargs)

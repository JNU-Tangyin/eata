"""
EATA-Simple: 简单奖励变体
替换复杂的Wasserstein距离为简单收益差
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.agent import Agent
import pandas as pd
import numpy as np

class EATASimple:
    """
    简单奖励的EATA变体
    通过替换Wasserstein距离为简单MAE测试复杂奖励机制的价值
    """
    
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        初始化简单奖励的EATA模型
        """
        self.name = "EATA-Simple"
        self.description = "简单奖励 - 替换复杂的Wasserstein距离为简单收益差"
        
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
            'reward_function': 'simple_mae',
            'target_file': 'agent.py',
            'target_line': 167,
            'modification_type': 'function_replacement'
        }
        
    def _apply_modifications(self):
        """
        应用消融修改：替换复杂奖励函数为简单MAE
        """
        try:
            # 重写Agent的奖励计算方法
            original_method = self.agent._calculate_rl_reward_and_signal
            
            def simple_reward_calculation(prediction_distribution, lookahead_ground_truth, shares_held):
                """
                简单奖励计算：使用MAE替代Wasserstein距离
                科学合理的简化版本，保持与原始方法的可比性
                """
                try:
                    if prediction_distribution.size == 0:
                        return 0.0, 0

                    # 交易信号决策（完全复制原始逻辑以保持可比性）
                    strategy = [25, 75]
                    q_low, q_high = np.percentile(prediction_distribution, strategy)
                    
                    print(f"  [简单调试] 预测分布: min={prediction_distribution.min():.6f}, max={prediction_distribution.max():.6f}")
                    print(f"  [简单调试] Q25={q_low:.6f}, Q75={q_high:.6f}, median={np.median(prediction_distribution):.6f}")
                    
                    intended_signal = 0
                    if q_low > 0:
                        intended_signal = 1
                        print(f"  [简单决策] 预测分布的 25% 分位数 > 0，生成意图信号: 买入")
                    elif q_high < 0:
                        intended_signal = -1
                        print(f"  [简单决策] 预测分布的 75% 分位数 < 0，生成意图信号: 卖出")
                    else:
                        if prediction_distribution.min() >= 0:
                            median_val = np.median(prediction_distribution)
                            threshold = (prediction_distribution.max() - prediction_distribution.min()) * 0.3
                            if median_val > threshold:
                                intended_signal = 1
                                print(f"  [简单决策] 全正分布，中位数{median_val:.6f} > 阈值{threshold:.6f}，生成意图信号: 买入")
                            else:
                                print(f"  [简单决策] 全正分布，中位数{median_val:.6f} <= 阈值{threshold:.6f}，生成意图信号: 持有")
                        else:
                            print("  [简单决策] 预测分布跨越零点，信号不明确，生成意图信号: 持有")

                    # RL奖励计算 - 关键修复：正确提取真实收益数据
                    actual_returns = lookahead_ground_truth.T[3, :]  # 与原始方法完全一致
                    
                    # 调试信息：检查输入数据
                    print(f"  [简单RL调试] 预测分布形状: {prediction_distribution.shape}, 范围: [{prediction_distribution.min():.6f}, {prediction_distribution.max():.6f}]")
                    print(f"  [简单RL调试] 真实收益形状: {actual_returns.shape}, 范围: [{actual_returns.min():.6f}, {actual_returns.max():.6f}]")
                    
                    # 检查输入数据有效性
                    if len(prediction_distribution) == 0 or len(actual_returns) == 0:
                        print(f"  ⚠️ 空的输入数据，返回默认RL奖励0.0")
                        return 0.0, intended_signal
                        
                    if np.all(np.isnan(prediction_distribution)) or np.all(np.isnan(actual_returns)):
                        print(f"  ⚠️ 输入数据全为nan，返回默认RL奖励0.0")
                        return 0.0, intended_signal
                    
                    # 简化的距离计算：使用MAE替代Wasserstein距离
                    # 这是统计学上合理的简化，MAE是L1距离，比Wasserstein距离计算简单但保持了分布比较的本质
                    simple_distance = np.mean(np.abs(prediction_distribution - np.mean(actual_returns)))
                    print(f"  [简单RL调试] MAE距离: {simple_distance}")
                    
                    # 处理异常的距离值
                    if np.isnan(simple_distance) or np.isinf(simple_distance):
                        print(f"  ⚠️ 异常的MAE距离: {simple_distance}")
                        print(f"  [简单诊断] 预测分布统计: mean={np.mean(prediction_distribution):.6f}, std={np.std(prediction_distribution):.6f}")
                        print(f"  [简单诊断] 真实收益统计: mean={np.mean(actual_returns):.6f}, std={np.std(actual_returns):.6f}")
                        return 0.0, intended_signal
                    elif simple_distance < 0:
                        print(f"  ⚠️ 负的MAE距离: {simple_distance}，这不应该发生")
                        return 0.0, intended_signal
                    
                    rl_reward = 1 / (1 + simple_distance)
                    print(f"  [简单RL调试] 计算的RL奖励: {rl_reward:.6f}")
                    
                    # 最终检查
                    if np.isnan(rl_reward) or np.isinf(rl_reward):
                        print(f"  ⚠️ 最终RL奖励异常: {rl_reward}，返回0.0")
                        rl_reward = 0.0
                    
                    return rl_reward, intended_signal
                    
                except Exception as e:
                    print(f"--- 🚨 简单奖励计算中捕获到错误 🚨 ---")
                    print(f"错误信息: {e}")
                    print(f"错误类型: {type(e).__name__}")
                    import traceback
                    print(f"完整错误堆栈:")
                    traceback.print_exc()
                    print(f"预测分布类型: {type(prediction_distribution)}, 形状: {getattr(prediction_distribution, 'shape', 'N/A')}")
                    print(f"真实数据类型: {type(lookahead_ground_truth)}, 形状: {getattr(lookahead_ground_truth, 'shape', 'N/A')}")
                    return 0.0, 0
            
            # 替换原方法
            self.agent._calculate_rl_reward_and_signal = simple_reward_calculation
            print(f"{self.name}: 已替换为简单MAE奖励函数")
            
        except Exception as e:
            print(f"{self.name}: 应用修改时出错: {e}")
    
    def run_backtest(self, train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
        """
        运行回测 - 使用与对比实验相同的核心回测逻辑
        """
        try:
            print(f"运行{self.name}回测 - {ticker}")
            print(f"   修改: reward_function='simple_mae' (简单奖励)")
            
            # 导入核心回测函数
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.insert(0, project_root)
            
            from predict import run_eata_core_backtest
            
            # 合并训练和测试数据
            combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
            
            # 使用核心回测函数，传入预配置的Agent（已经替换了奖励函数）
            core_metrics, portfolio_df = run_eata_core_backtest(
                stock_df=combined_df,
                ticker=ticker,
                lookback=50,
                lookahead=10,
                stride=1,
                depth=300,
                variant_params=None,  # Simple变体不使用参数传递
                pre_configured_agent=self.agent  # 使用已经修改过奖励函数的Agent
            )
            print(f"  [调试] 核心回测函数执行完成")
            print(f"  [调试] 返回的指标: {core_metrics}")
            
            # 提取指标
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
            
            print(f"{self.name}回测完成 - 年化收益: {annual_return:.4f}, RL奖励: {avg_rl_reward:.6f}")
            
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
            'hypothesis': '对分布的鲁棒性变差，容易受到极端行情噪声点影响',
            'expected_performance': {
                'distribution_robustness': '-60%',
                'extreme_event_handling': 'poor',
                'noise_sensitivity': 'high'
            }
        }

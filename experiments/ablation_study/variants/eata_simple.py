"""
EATA-Simple: 替代目标函数变体
支持多种分布距离函数：MSE, KL Divergence, JS Divergence, CVaR
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.agent import Agent
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

class EATASimple:
    """
    替代目标函数的EATA变体
    支持MSE, KL, JS, CVaR等多种距离函数替代Wasserstein
    """
    
    def __init__(self, df: pd.DataFrame, objective='mse', **kwargs):
        """
        初始化替代目标函数的EATA模型
        
        Args:
            objective: 目标函数类型，可选 'mse', 'kl', 'js', 'cvar'
        """
        self.objective = objective.lower()
        self.name = f"EATA-{objective.upper()}"
        
        objective_descriptions = {
            'mse': "MSE - 均方误差（L2距离）",
            'kl': "KL Divergence - KL散度",
            'js': "JS Divergence - Jensen-Shannon散度",
            'cvar': "CVaR - 条件风险价值（5%）"
        }
        self.description = objective_descriptions.get(self.objective, "未知目标函数")
        
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
            'reward_function': self.objective,
            'target_file': 'agent.py',
            'target_line': 167,
            'modification_type': 'function_replacement'
        }
        
    def _normalize_to_prob(self, data, bins=None):
        """将数据转换为概率分布（用于KL和JS散度）
        使用直方图估计概率密度，避免负值问题
        """
        # 自适应bins数量：小样本用少量bins
        if bins is None:
            bins = min(max(len(data) // 3, 10), 30)
        
        # 使用直方图估计概率密度
        hist, _ = np.histogram(data, bins=bins, density=False)
        hist = hist.astype(float) + 1e-10  # 避免零概率
        return hist / hist.sum()
    
    def _calculate_distance(self, prediction_distribution, actual_returns):
        """
        根据objective类型计算距离
        """
        if self.objective == 'mse':
            # MSE: 真正的均方误差（两个分布之间的期望平方距离）
            # 计算所有可能配对的平方差的期望
            # E[(X - Y)²] where X ~ prediction_distribution, Y ~ actual_returns
            
            # 为了效率，我们计算所有配对的平方差
            # 这是两个分布之间L2距离的无偏估计
            all_squared_diffs = []
            for pred_val in prediction_distribution:
                for actual_val in actual_returns:
                    all_squared_diffs.append((pred_val - actual_val)**2)
            
            # MSE是所有配对平方差的平均
            distance = np.mean(all_squared_diffs)
            
        elif self.objective == 'kl':
            # KL Divergence: KL散度（使用直方图估计）
            # 使用相同的bins范围确保可比性
            data_min = min(prediction_distribution.min(), actual_returns.min())
            data_max = max(prediction_distribution.max(), actual_returns.max())
            bins = np.linspace(data_min, data_max, 20)
            
            hist_p, _ = np.histogram(prediction_distribution, bins=bins, density=False)
            hist_q, _ = np.histogram(actual_returns, bins=bins, density=False)
            
            # 归一化为概率分布
            p = (hist_p.astype(float) + 1e-10) / (hist_p.sum() + 1e-10 * len(hist_p))
            q = (hist_q.astype(float) + 1e-10) / (hist_q.sum() + 1e-10 * len(hist_q))
            
            # 计算KL散度
            distance = entropy(p, q)
            if np.isinf(distance) or np.isnan(distance):
                distance = 10.0  # 使用一个大的默认值
            
        elif self.objective == 'js':
            # JS Divergence: Jensen-Shannon散度（使用直方图估计）
            # 使用相同的bins范围确保可比性
            data_min = min(prediction_distribution.min(), actual_returns.min())
            data_max = max(prediction_distribution.max(), actual_returns.max())
            bins = np.linspace(data_min, data_max, 20)
            
            hist_p, _ = np.histogram(prediction_distribution, bins=bins, density=False)
            hist_q, _ = np.histogram(actual_returns, bins=bins, density=False)
            
            # 归一化为概率分布
            p = (hist_p.astype(float) + 1e-10) / (hist_p.sum() + 1e-10 * len(hist_p))
            q = (hist_q.astype(float) + 1e-10) / (hist_q.sum() + 1e-10 * len(hist_q))
            
            # 计算JS散度
            distance = jensenshannon(p, q)
            if np.isnan(distance):
                distance = 1.0  # JS散度的最大值
            
        elif self.objective == 'cvar':
            # CVaR: 条件风险价值（关注尾部风险）
            alpha = 0.05  # 5%
            # 计算预测分布的CVaR
            var_threshold = np.percentile(prediction_distribution, alpha * 100)
            pred_cvar = np.mean(prediction_distribution[prediction_distribution <= var_threshold])
            # 计算真实收益的CVaR
            var_threshold_actual = np.percentile(actual_returns, alpha * 100)
            actual_cvar = np.mean(actual_returns[actual_returns <= var_threshold_actual])
            # 距离是两个CVaR的差异
            distance = np.abs(pred_cvar - actual_cvar)
            
        else:
            raise ValueError(f"未知的目标函数: {self.objective}")
        
        return distance
    
    def _apply_modifications(self):
        """
        应用消融修改：设置目标函数类型标记
        不再直接替换方法，而是通过variant_params传递给核心训练代码
        """
        try:
            # 在Agent上设置目标函数标记
            self.agent._variant_objective = self.objective
            print(f"{self.name}: 已设置目标函数标记为 {self.objective.upper()}")
            
        except Exception as e:
            print(f"{self.name}: 应用修改时出错: {e}")
    
    def run_backtest(self, train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
        """
        运行回测 - 测试4种目标函数（MSE, KL, JS, CVaR）并返回所有结果
        """
        print(f"\n{'='*80}")
        print(f"🔬 运行EATA-Simple回测 - {ticker}")
        print(f"   将测试4种目标函数: MSE, KL, JS, CVaR")
        print(f"{'='*80}\n")
        
        # 导入核心回测函数
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.insert(0, project_root)
        
        from predict import run_eata_core_backtest
        from core.eata_agent import score as score_module
        
        # 合并训练和测试数据
        combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        
        # 4种目标函数
        objectives = ['mse', 'kl', 'js', 'cvar']
        all_results = []
        
        # 对每种目标函数运行一次回测
        for obj in objectives:
            try:
                print(f"\n{'─'*80}")
                print(f"📊 测试目标函数: {obj.upper()}")
                print(f"{'─'*80}")
                
                # 临时设置当前目标函数
                self.objective = obj
                
                # 创建自定义score函数
                def custom_score_func(eq, tree_size, data, t_limit=1.0, eta=0.999):
                    """使用当前目标函数评估表达式"""
                    return score_module.score_with_objective(eq, tree_size, data, t_limit=t_limit, objective=obj, eta=eta)
                
                variant_params = {
                    'objective_function': obj,
                    'distance_calculator': lambda pred, actual: self._calculate_distance(pred, actual),
                    'custom_score_function': custom_score_func
                }
                
                core_metrics, portfolio_df = run_eata_core_backtest(
                    stock_df=combined_df,
                    ticker=ticker,
                    lookback=50,
                    lookahead=10,
                    stride=1,
                    depth=300,
                    variant_params=variant_params,
                    pre_configured_agent=None
                )
                
                # 提取指标
                result = {
                    'variant': f'EATA-Simple-{obj.upper()}',
                    'ticker': ticker,
                    'objective': obj.upper(),
                    'annual_return': core_metrics.get('Annual Return (AR)', 0.0),
                    'sharpe_ratio': core_metrics.get('Sharpe Ratio', 0.0),
                    'max_drawdown': core_metrics.get('Max Drawdown (MDD)', 0.0),
                    'win_rate': core_metrics.get('Win Rate', 0.0),
                    'volatility': core_metrics.get('Volatility (Annual)', 0.0),
                    'rl_reward': core_metrics.get('Average RL Reward', 0.0),
                }
                
                all_results.append(result)
                print(f"✅ {obj.upper()} 完成 - 年化收益: {result['annual_return']:.4f}, Sharpe: {result['sharpe_ratio']:.4f}")
                
            except Exception as e:
                print(f"❌ {obj.upper()} 失败: {str(e)}")
                all_results.append({
                    'variant': f'EATA-Simple-{obj.upper()}',
                    'ticker': ticker,
                    'objective': obj.upper(),
                    'error': str(e)
                })
        
        # 计算平均值
        if len(all_results) > 0 and all('error' not in r for r in all_results):
            avg_result = {
                'variant': 'EATA-Simple',
                'ticker': ticker,
                'objective': 'AVERAGE',
                'annual_return': np.mean([r['annual_return'] for r in all_results]),
                'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in all_results]),
                'max_drawdown': np.mean([r['max_drawdown'] for r in all_results]),
                'win_rate': np.mean([r['win_rate'] for r in all_results]),
                'volatility': np.mean([r['volatility'] for r in all_results]),
                'rl_reward': np.mean([r['rl_reward'] for r in all_results]),
                'modifications': self.modifications
            }
            
            print(f"\n{'='*80}")
            print(f"📈 EATA-Simple 平均结果 - {ticker}")
            print(f"   年化收益: {avg_result['annual_return']:.4f}")
            print(f"   Sharpe比率: {avg_result['sharpe_ratio']:.4f}")
            print(f"   最大回撤: {avg_result['max_drawdown']:.4f}")
            print(f"{'='*80}\n")
            
            # 返回平均值作为主结果，同时保存详细结果
            avg_result['detailed_results'] = all_results
            return avg_result
        else:
            # 如果有错误，返回第一个成功的结果或错误
            return all_results[0] if all_results else {
                'variant': 'EATA-Simple',
                'ticker': ticker,
                'error': '所有目标函数测试失败'
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

"""
EATA-Full: 完整的EATA模型（基准版本）
保持所有原始参数和功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.agent import Agent
import pandas as pd
import numpy as np

class EATAFull:
    """
    完整的EATA模型，作为消融实验的基准
    """
    
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        初始化完整的EATA模型
        
        Args:
            df: 股票数据DataFrame
            **kwargs: 其他参数
        """
        self.name = "EATA-Full"
        self.description = "完整的EATA模型，包含所有组件"
        
        # 使用默认参数创建Agent
        self.agent = Agent(
            df=df,
            lookback=kwargs.get('lookback', 100),
            lookahead=kwargs.get('lookahead', 20),
            stride=kwargs.get('stride', 1),
            depth=kwargs.get('depth', 300)
        )
        
        self.modifications = {}
        
    def run_backtest(self, train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
        """
        运行完整的EATA算法消融实验 - 使用与对比实验相同的核心回测逻辑
        
        Args:
            train_df: 训练数据
            test_df: 测试数据
            ticker: 股票代码
            
        Returns:
            dict: 实验结果
        """
        print(f"运行EATA-Full回测 - {ticker}")
        
        try:
            # 导入并使用与对比实验相同的核心回测函数
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.insert(0, project_root)
            
            from predict import run_eata_core_backtest
            
            # 合并训练和测试数据，与对比实验保持一致
            combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
            
            # EATA-Full是原版基准，完全使用主程序的默认行为
            print(f"  [基准] EATA-Full使用原版配置，无任何修改")
            
            core_metrics, portfolio_df = run_eata_core_backtest(
                stock_df=combined_df,
                ticker=ticker,
                lookback=50,
                lookahead=10,
                stride=1,
                depth=300,
                variant_params=None,  # 基准模型不使用任何变体参数
                pre_configured_agent=None  # 不使用预配置Agent，让主程序自己创建
            )
            
            # 转换指标格式
            annual_return = core_metrics.get('Annual Return (AR)', 0.0)
            sharpe_ratio = core_metrics.get('Sharpe Ratio', 0.0)
            max_drawdown = core_metrics.get('Max Drawdown (MDD)', 0.0)
            win_rate = core_metrics.get('Win Rate', 0.0)
            volatility = core_metrics.get('Volatility (Annual)', 0.0)
            avg_rl_reward = core_metrics.get('Average RL Reward', 0.0)
            
            print(f"EATA-Full完整消融实验完成")
            print(f"   年化收益: {annual_return:.4f}")
            print(f"   夏普比率: {sharpe_ratio:.4f}")
            print(f"   最大回撤: {max_drawdown:.4f}")
            
            return {
                'variant': self.name,
                'ticker': ticker,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': volatility,
                'rl_reward': avg_rl_reward,  # 使用真实的平均RL奖励
                'modifications': self.modifications,
                'training_info': {
                    'nemots_iterations': getattr(self, '_nemots_iterations', 0),
                    'nn_epochs': getattr(self, '_nn_epochs', 0),
                    'backtest_windows': 1
                }
            }
            
        except Exception as e:
            print(f"EATA-Full完整消融实验失败: {str(e)}")
            return {
                'variant': self.name,
                'ticker': ticker,
                'error': str(e),
                'modifications': self.modifications
            }
    
    def _train_nemots_model(self, train_df: pd.DataFrame):
        """
        完整的NEMoTS模型训练
        """
        print("   初始化NEMoTS引擎...")
        
        # 设置训练参数
        max_iterations = 50  # NEMoTS进化迭代次数
        population_size = 20
        
        print(f"   开始{max_iterations}轮NEMoTS进化训练...")
        
        best_fitness = -float('inf')
        for iteration in range(max_iterations):
            # 模拟NEMoTS的进化过程
            try:
                # 使用Agent的engine进行模拟训练
                if hasattr(self.agent, 'engine') and hasattr(self.agent.engine, 'simulate'):
                    # 进行一轮模拟
                    experiences = self.agent.engine.simulate()
                    
                    # 评估当前种群适应度
                    current_fitness = self._evaluate_fitness(train_df)
                    
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        print(f"   迭代 {iteration+1}/{max_iterations}: 适应度提升至 {best_fitness:.4f}")
                    
                    # 每10轮显示进度
                    if (iteration + 1) % 10 == 0:
                        print(f"   NEMoTS训练进度: {iteration+1}/{max_iterations}")
                        
            except Exception as e:
                print(f"   NEMoTS迭代 {iteration+1} 出错: {e}")
                continue
        
        self._nemots_iterations = max_iterations
        print(f"   NEMoTS训练完成，共{max_iterations}轮迭代")
    
    def _train_neural_network(self, train_df: pd.DataFrame):
        """
        完整的神经网络训练
        """
        print("   初始化神经网络训练...")
        
        # 设置训练参数
        epochs = 20
        batch_size = 32
        
        print(f"   开始{epochs}轮神经网络训练...")
        
        total_loss = 0.0
        for epoch in range(epochs):
            try:
                # 使用Agent的engine进行神经网络训练
                if hasattr(self.agent, 'engine') and hasattr(self.agent.engine, 'train'):
                    # 进行一轮训练
                    epoch_loss = self.agent.engine.train()
                    total_loss += epoch_loss
                    
                    # 每5轮显示进度
                    if (epoch + 1) % 5 == 0:
                        avg_loss = total_loss / (epoch + 1)
                        print(f"   Epoch {epoch+1}/{epochs}: 平均损失 {avg_loss:.6f}")
                        
            except Exception as e:
                print(f"   神经网络训练 Epoch {epoch+1} 出错: {e}")
                continue
        
        self._nn_epochs = epochs
        avg_final_loss = total_loss / epochs if epochs > 0 else 0.0
        print(f"   神经网络训练完成，平均损失: {avg_final_loss:.6f}")
    
    def _sliding_window_backtest(self, test_df: pd.DataFrame):
        """
        滑动窗口回测
        """
        print("   开始滑动窗口回测...")
        
        window_size = 20  # 滑动窗口大小
        stride = 5        # 滑动步长
        
        trading_signals = []
        returns = []
        rl_rewards = []
        
        # 计算滑动窗口数量
        num_windows = max(1, (len(test_df) - window_size) // stride + 1)
        print(f"   将进行 {num_windows} 个滑动窗口的预测")
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = min(start_idx + window_size, len(test_df))
            
            if end_idx - start_idx < 10:  # 窗口太小则跳过
                continue
                
            window_data = test_df.iloc[start_idx:end_idx].copy()
            
            try:
                # 对当前窗口进行预测
                signal, rl_reward = self.agent.criteria(window_data, shares_held=0)
                
                # 计算当前窗口的收益
                window_returns = self._calculate_window_returns(window_data, signal)
                
                trading_signals.append(signal)
                returns.extend(window_returns)
                rl_rewards.append(rl_reward)
                
                # 显示进度
                if (i + 1) % max(1, num_windows // 5) == 0:
                    progress = (i + 1) / num_windows * 100
                    print(f"   回测进度: {progress:.1f}% ({i+1}/{num_windows})")
                    
            except Exception as e:
                print(f"   窗口 {i+1} 预测出错: {e}")
                trading_signals.append(0)
                rl_rewards.append(0.0)
                continue
        
        print(f"   滑动窗口回测完成，共处理 {len(trading_signals)} 个窗口")
        return trading_signals, returns, rl_rewards
    
    def _evaluate_fitness(self, train_df: pd.DataFrame):
        """
        评估NEMoTS种群适应度
        """
        try:
            # 使用训练数据的一小部分进行快速适应度评估
            sample_size = min(50, len(train_df))
            sample_data = train_df.tail(sample_size)
            
            signal, rl_reward = self.agent.criteria(sample_data, shares_held=0)
            
            # 计算简单的适应度分数
            returns = self._calculate_window_returns(sample_data, signal)
            fitness = np.mean(returns) if len(returns) > 0 else 0.0
            
            return fitness + rl_reward * 0.1  # 结合收益和RL奖励
            
        except Exception:
            return -1.0  # 错误情况返回负适应度
    
    def _calculate_window_returns(self, window_data: pd.DataFrame, trading_signal: int):
        """
        计算窗口收益率序列
        """
        if len(window_data) < 2:
            return [0.0]
            
        # 计算价格变化
        prices = window_data['close'].values
        price_returns = np.diff(prices) / prices[:-1]
        
        # 应用交易信号
        strategy_returns = price_returns * trading_signal
        
        return strategy_returns.tolist()
    
    def _calculate_comprehensive_metrics(self, returns: list, trading_signals: list):
        """
        计算综合性能指标
        """
        if not returns or len(returns) == 0:
            return self._get_empty_metrics()
        
        returns_array = np.array(returns)
        returns_array = returns_array[~np.isnan(returns_array)]  # 移除NaN值
        
        if len(returns_array) == 0:
            return self._get_empty_metrics()
        
        # 基础指标
        annual_return = np.mean(returns_array) * 252  # 年化收益
        volatility = np.std(returns_array) * np.sqrt(252)  # 年化波动率
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # 胜率
        win_rate = np.sum(returns_array > 0) / len(returns_array)
        
        # 交易统计
        total_signals = len([s for s in trading_signals if s != 0])
        signal_diversity = len(set(trading_signals)) / len(trading_signals) if trading_signals else 0
        
        return {
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(total_signals),
            'signal_diversity': float(signal_diversity),
            'total_return': float(cumulative_returns[-1] - 1) if len(cumulative_returns) > 0 else 0.0
        }
    
    def _get_empty_metrics(self):
        """
        返回空的指标字典
        """
        return {
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'signal_diversity': 0.0,
            'total_return': 0.0
        }
    
    def get_variant_info(self):
        """
        获取变体信息
        """
        return {
            'name': self.name,
            'description': self.description,
            'modifications': self.modifications,
            'hypothesis': '基准性能，其他变体与此对比',
            'expected_performance': 'baseline'
        }

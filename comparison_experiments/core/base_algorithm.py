"""
统一的算法基类接口

所有参与对比的算法都需要继承这个基类，确保接口一致性。
"""

import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List


class BaseAlgorithm(ABC):
    """算法基类，定义统一接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化算法
        
        Args:
            config: 算法配置参数
        """
        self.config = config
        self.name = self.__class__.__name__.replace('Algorithm', '')
        self.is_trained = False
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        训练算法
        
        Args:
            train_data: 训练数据
        """
        pass
    
    @abstractmethod
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            test_data: 测试数据
            
        Returns:
            预测结果 (交易信号或价格预测)
        """
        pass
    
    def backtest(self, data: pd.DataFrame, initial_cash: float = 1000000) -> Tuple[List[float], List[Dict]]:
        """
        回测算法性能
        
        Args:
            data: 回测数据
            initial_cash: 初始资金
            
        Returns:
            (portfolio_values, trades): 组合价值序列和交易记录
        """
        portfolio_values = [initial_cash]
        trades = []
        cash = initial_cash
        position = 0  # 持仓数量
        
        # 获取预测信号
        predictions = self.predict(data)
        
        for i, (idx, row) in enumerate(data.iterrows()):
            if i >= len(predictions):
                break
                
            current_price = row['close']
            signal = predictions[i]
            
            # 简单的交易逻辑：信号>0买入，信号<0卖出
            if signal > 0 and position == 0:  # 买入信号且无持仓
                shares_to_buy = int(cash / current_price)
                if shares_to_buy > 0:
                    position = shares_to_buy
                    cash -= shares_to_buy * current_price
                    trades.append({
                        'date': idx,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'cash_after': cash
                    })
            elif signal < 0 and position > 0:  # 卖出信号且有持仓
                cash += position * current_price
                trades.append({
                    'date': idx,
                    'action': 'SELL', 
                    'price': current_price,
                    'shares': position,
                    'cash_after': cash
                })
                position = 0
            
            # 计算当前组合价值
            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)
        
        return portfolio_values, trades
    
    def evaluate(self, data: pd.DataFrame, initial_cash: float = 1000000) -> Dict[str, float]:
        """
        评估算法性能
        
        Args:
            data: 评估数据
            initial_cash: 初始资金
            
        Returns:
            性能指标字典
        """
        start_time = time.time()
        
        # 如果模型未训练，先进行训练
        if not self.is_trained:
            self.fit(data)
        
        # 执行回测
        portfolio_values, trades = self.backtest(data, initial_cash)
        
        # 计算性能指标
        metrics = self._calculate_metrics(portfolio_values, initial_cash)
        
        # 添加算法特定信息
        metrics.update({
            'algorithm': self.name,
            'total_time': time.time() - start_time,
            'num_trades': len(trades),
            'config': self.config.copy()
        })
        
        return metrics
    
    def _calculate_metrics(self, portfolio_values: List[float], initial_cash: float) -> Dict[str, float]:
        """计算标准化性能指标"""
        if len(portfolio_values) < 2:
            return self._get_zero_metrics()
        
        # 转换为pandas Series便于计算
        values = pd.Series(portfolio_values)
        returns = values.pct_change().dropna()
        
        # 基本收益指标
        total_return = (values.iloc[-1] - initial_cash) / initial_cash
        
        # 年化收益率 (假设252个交易日/年)
        num_periods = len(values) - 1
        years = num_periods / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # 风险指标
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cumulative = values / initial_cash
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = drawdown.max()
        
        # 波动率
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_portfolio_value': values.iloc[-1],
            'num_periods': num_periods
        }
    
    def _get_zero_metrics(self) -> Dict[str, float]:
        """返回零值指标（用于失败情况）"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'final_portfolio_value': 0.0,
            'num_periods': 0
        }
    
    def __str__(self) -> str:
        return f"{self.name}({self.config})"
    
    def __repr__(self) -> str:
        return self.__str__()

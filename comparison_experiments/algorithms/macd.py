"""
MACD 策略
基于MACD指标的交叉策略
"""

import pandas as pd
import numpy as np

try:
    from .data_utils import run_vectorized_backtest
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from data_utils import run_vectorized_backtest


def run_macd_strategy(df: pd.DataFrame):
    """
    MACD交叉策略
    
    Args:
        df: 包含MACD指标的DataFrame
        
    Returns:
        tuple: (metrics, backtest_results)
    """
    print("Running MACD Crossover strategy...")
    
    df_macd = df.copy()
    
    # 确保MACD列存在
    if 'macd_12_26_9' not in df_macd.columns or 'macds_12_26_9' not in df_macd.columns:
        raise ValueError("MACD and MACDS columns are required for this strategy.")
    
    # MACD > Signal Line = 买入信号
    df_macd['signal'] = (df_macd['macd_12_26_9'] > df_macd['macds_12_26_9']).astype(int)
    
    metrics, backtest_results = run_vectorized_backtest(df_macd, signal_col='signal')
    
    print(f"✅ MACD strategy completed")
    print(f"   Total Return: {metrics['total_return']:.2%}")
    print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    return metrics, backtest_results


if __name__ == '__main__':
    # 测试代码
    from data_utils import add_technical_indicators
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    prices = [100]
    for _ in range(len(dates)-1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    
    df = pd.DataFrame({
        'date': dates,
        'ticker': 'TEST',
        'open': [p * 0.99 for p in prices],
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    df = add_technical_indicators(df)
    metrics, _ = run_macd_strategy(df)
    print("MACD strategy test completed!")

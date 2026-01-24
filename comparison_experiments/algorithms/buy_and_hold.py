"""
Buy and Hold 策略
最简单的基准策略，始终持有股票
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


def run_buy_hold_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    """
    买入持有策略
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        ticker: 股票代码
        
    Returns:
        tuple: (metrics, backtest_results)
    """
    print(f"Running Buy and Hold strategy for {ticker}...")
    
    df_bh = test_df.copy()
    df_bh['signal'] = 1  # 始终持有
    
    metrics, backtest_results = run_vectorized_backtest(df_bh, signal_col='signal')
    
    print(f"✅ Buy and Hold completed")
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
    metrics, _ = run_buy_and_hold(df)
    print("Buy and Hold strategy test completed!")

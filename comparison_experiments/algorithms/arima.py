"""
ARIMA 策略
基于ARIMA时间序列预测的交易策略
"""

import pandas as pd
import numpy as np
import warnings

# 忽略 urllib3 在 LibreSSL 环境下关于 NotOpenSSL 的兼容性提示
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# 禁用statsmodels相关警告
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
# 禁用statsmodels ConvergenceWarning
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization.*")
warnings.filterwarnings("ignore", message=".*Check mle_retvals.*")

try:
    from .data_utils import run_vectorized_backtest
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from data_utils import run_vectorized_backtest


def run_arima_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    """
    ARIMA策略 - 基于ETS-SDA原始实现
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        ticker: 股票代码
        
    Returns:
        tuple: (metrics, backtest_results)
    """
    print(f"Running ARIMA strategy for {ticker}...")
    
    try:
        # 延迟导入避免死锁，重定向stdout来隐藏darts的导入信息
        import sys
        from io import StringIO
        
        # 临时重定向stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            from darts import TimeSeries
            from darts.models import ARIMA
        finally:
            # 恢复stdout
            sys.stdout = old_stdout
            
        import numpy as np

        # 忽略 statsmodels 在 ARIMA 初始参数上的非平稳/不可逆警告
        warnings.filterwarnings(
            "ignore",
            message="Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Non-invertible starting MA parameters found. Using zeros as starting parameters.",
            category=UserWarning,
        )
        
        # 1. 准备数据 - 按照ETS-SDA的方式
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
        train_series = TimeSeries.from_dataframe(train_df, 'date', 'close', freq='B')
        test_series = TimeSeries.from_dataframe(test_df, 'date', 'close', freq='B')
        
        # 2. 训练ARIMA模型 - 改进参数
        print("   Training ARIMA model...")
        
        # 尝试多个ARIMA参数组合，选择最佳的
        best_model = None
        best_aic = float('inf')
        
        param_combinations = [
            (1, 1, 1), (2, 1, 2), (3, 1, 3), (5, 1, 0),  # 包含原始参数
            (1, 1, 0), (0, 1, 1), (2, 1, 1), (1, 1, 2)
        ]
        
        for p, d, q in param_combinations:
            try:
                temp_model = ARIMA(p=p, d=d, q=q)
                temp_model.fit(train_series)
                aic = temp_model.model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_model = temp_model
                    print(f"   Better model found: ARIMA({p},{d},{q}) with AIC={aic:.2f}")
            except:
                continue
        
        if best_model is None:
            # 如果所有参数都失败，使用简单的(1,1,1)
            best_model = ARIMA(p=1, d=1, q=1)
            best_model.fit(train_series)
        
        model = best_model
        
        # 3. 预测 - 按照ETS-SDA的方式
        print("   Making predictions...")
        predictions = model.predict(n=len(test_series), series=train_series)
        
        # 4. 生成信号 - 按照ETS-SDA的逻辑
        df_arima = test_df.copy()
        # 对齐预测值与测试集索引
        predicted_values = predictions.values().flatten()
        df_arima['predicted_close'] = predicted_values[-len(test_df):]
        
        # 信号生成：如果预测价格高于当前价格则买入，否则卖出
        df_arima['signal'] = np.where(
            df_arima['predicted_close'] > df_arima['close'].shift(-1), 1, -1
        )
        df_arima['signal'] = df_arima['signal'].fillna(0)  # 最后一个信号填充为0（持有）
        
        # 5. 运行回测
        metrics, backtest_results = run_vectorized_backtest(df_arima, signal_col='signal')
        
        print(f"✅ ARIMA strategy completed for {ticker}")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        return metrics, backtest_results
        
    except Exception as e:
        print(f"❌ ARIMA strategy failed for {ticker}: {e}")
        # 返回默认结果
        performance_metrics = pd.Series({
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'num_trades': 0
        })
        equity_curve = pd.DataFrame({
            'date': test_df['date'], 
            'portfolio_value': 1.0
        })
        return performance_metrics, equity_curve


if __name__ == '__main__':
    # 测试代码
    from data_utils import add_technical_indicators
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
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
    
    # 分割数据
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    try:
        metrics, _ = run_arima_strategy(train_df, test_df, 'TEST')
        print("ARIMA strategy test completed!")
    except Exception as e:
        print(f"ARIMA test failed: {e}")
        print("This is expected if darts is not installed properly")

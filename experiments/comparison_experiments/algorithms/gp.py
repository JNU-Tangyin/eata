"""
Genetic Programming 策略
使用遗传编程进化交易规则
"""

import pandas as pd
import numpy as np
import warnings

# 禁用sklearn FutureWarning
warnings.filterwarnings("ignore", message="`BaseEstimator._validate_data` is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

try:
    from .data_utils import run_vectorized_backtest
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from data_utils import run_vectorized_backtest


def run_gp_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    """
    遗传编程策略 - 使用gplearn实现真正的遗传编程
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        ticker: 股票代码
        
    Returns:
        tuple: (metrics, backtest_results)
    """
    print(f"Running Genetic Programming strategy for {ticker}...")
    
    try:
        # 修复gplearn与sklearn 1.6.1的兼容性问题
        import sklearn.utils.validation
        import sklearn.base
        
        # 为gplearn添加缺失的_validate_data方法
        if not hasattr(sklearn.base.BaseEstimator, '_validate_data'):
            def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
                from sklearn.utils.validation import check_X_y, check_array
                if y is not None:
                    X, y = check_X_y(X, y, **check_params)
                    return X, y
                else:
                    X = check_array(X, **check_params)
                    return X
            
            # 动态添加方法到sklearn.base.BaseEstimator
            sklearn.base.BaseEstimator._validate_data = _validate_data
        
        # 修复n_features_in_属性问题
        def patched_fit(original_fit):
            def wrapper(self, X, y, sample_weight=None):
                result = original_fit(self, X, y, sample_weight)
                # 添加缺失的属性
                if not hasattr(self, 'n_features_in_'):
                    self.n_features_in_ = X.shape[1]
                return result
            return wrapper
        
        # 现在可以安全导入gplearn
        from gplearn.genetic import SymbolicRegressor
        import numpy as np
        
        # 应用补丁到SymbolicRegressor
        if hasattr(SymbolicRegressor, 'fit') and not hasattr(SymbolicRegressor, '_original_fit'):
            SymbolicRegressor._original_fit = SymbolicRegressor.fit
            SymbolicRegressor.fit = patched_fit(SymbolicRegressor._original_fit)
        
        # 1. 准备数据 - 按照ETS-SDA的方式
        # 删除技术指标引入的NaN值
        train_df = train_df.dropna()
        
        # 选择特征列（排除基本价格和日期列）
        feature_cols = [col for col in train_df.columns if
                        col not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        
        X_train = train_df[feature_cols].values
        
        # 创建目标变量（从清理后的数据框）
        y_train = (train_df['close'].shift(-1) - train_df['close']).values
        
        # 清理X和y，确保没有NaN或Inf传递给模型
        X_train = np.nan_to_num(X_train)
        y_train = np.nan_to_num(y_train)
        
        # 准备测试数据，确保也是干净的
        X_test = test_df[feature_cols].values
        X_test = np.nan_to_num(X_test)
        
        # 2. 训练符号回归模型 - 使用ETS-SDA的参数
        print("   Training GP model...")
        gp_model = SymbolicRegressor(
            population_size=100,  # 减少以加快执行
            generations=3,        # 减少以加快执行
            stopping_criteria=0.01,
            p_crossover=0.7, 
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05, 
            p_point_mutation=0.1,
            max_samples=0.9, 
            verbose=0,
            parsimony_coefficient=0.01, 
            random_state=0
        )
        
        # 确保数据格式正确
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        # 移除任何剩余的无效值
        valid_indices = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train) | 
                         np.isinf(X_train).any(axis=1) | np.isinf(y_train))
        X_train_clean = X_train[valid_indices]
        y_train_clean = y_train[valid_indices]
        
        if len(X_train_clean) < 10:  # 数据不足
            raise ValueError("Insufficient clean training data for GP model")
            
        gp_model.fit(X_train_clean, y_train_clean)
        
        # 3. 预测
        print("   Making predictions...")
        predictions = gp_model.predict(X_test)
        
        # 4. 生成信号 - 基于价格变化预测
        df_gp = test_df.copy()
        df_gp['predicted_change'] = predictions
        
        # 根据预测的价格变化生成信号
        df_gp['signal'] = np.where(df_gp['predicted_change'] > 0, 1, -1)
        
        # 5. 运行回测
        metrics, backtest_results = run_vectorized_backtest(df_gp, signal_col='signal')
        
        print(f"✅ GP strategy completed for {ticker}")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   GP Formula: {str(gp_model._program)[:100]}...")
        
        return metrics, backtest_results
        
    except Exception as e:
        print(f"❌ GP strategy failed for {ticker}: {e}")
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
        metrics, _ = run_gp_strategy(train_df, test_df, 'TEST')
        print("GP strategy test completed!")
    except Exception as e:
        print(f"GP test failed: {e}")
        print("This is expected if gplearn is not installed properly")

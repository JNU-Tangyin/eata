"""
LightGBM 策略
使用LightGBM进行价格预测的交易策略
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 解决LightGBM环境冲突问题
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LightGBM导入失败: {e}")
    print("   尝试使用替代的梯度提升实现...")
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        LIGHTGBM_AVAILABLE = False
        SKLEARN_GB_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False
        SKLEARN_GB_AVAILABLE = False

try:
    from .data_utils import run_vectorized_backtest
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from data_utils import run_vectorized_backtest


def run_lightgbm_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    """
    LightGBM策略 - 使用原生LightGBM进行价格预测
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        ticker: 股票代码
        
    Returns:
        tuple: (metrics, backtest_results)
    """
    print(f"Running LightGBM strategy for {ticker}...")
    
    try:
        print("   Training LightGBM model...")
        
        # 1. 数据预处理
        def preprocess_data(df):
            df = df.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            for col in df.columns:
                if df[col].dtype == 'object' and col not in ['ticker', 'date']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
            return df.dropna()

        train_df = preprocess_data(train_df)
        test_df = preprocess_data(test_df)

        # 2. 特征工程
        feature_cols = [col for col in train_df.columns if
                        col not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        
        # 如果没有技术指标，创建基本特征
        if len(feature_cols) == 0:
            train_df['price_change'] = train_df['close'].pct_change().fillna(0)
            train_df['volume_change'] = train_df['volume'].pct_change().fillna(0)
            feature_cols = ['price_change', 'volume_change']
            
            test_df['price_change'] = test_df['close'].pct_change().fillna(0)
            test_df['volume_change'] = test_df['volume'].pct_change().fillna(0)
        
        # 创建滞后特征
        for lag in [1, 2, 3, 5]:
            train_df[f'close_lag_{lag}'] = train_df['close'].shift(lag)
            test_df[f'close_lag_{lag}'] = test_df['close'].shift(lag)
            feature_cols.append(f'close_lag_{lag}')
        
        # 删除包含NaN的行
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
        # 3. 准备训练数据
        X_train = train_df[feature_cols].values
        y_train = train_df['close'].shift(-1).fillna(train_df['close']).values[:-1]  # 预测下一期价格
        X_train = X_train[:-1]  # 对齐长度
        
        X_test = test_df[feature_cols].values
        
        # 4. 标准化特征
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 5. 训练梯度提升模型
        if LIGHTGBM_AVAILABLE:
            print("   使用原生LightGBM...")
            train_data = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
        elif SKLEARN_GB_AVAILABLE:
            print("   使用sklearn GradientBoostingRegressor...")
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                subsample=0.8
            )
            model.fit(X_train, y_train)
        else:
            raise ImportError("无法导入任何梯度提升库")
        
        # 6. 预测
        print("   Making predictions...")
        if LIGHTGBM_AVAILABLE:
            predictions = model.predict(X_test)
        else:
            # sklearn模型
            predictions = model.predict(X_test)
        
        # 7. 生成交易信号
        df_lgbm = test_df.copy()
        df_lgbm['predicted_close'] = predictions
        
        # 如果预测价格高于当前价格则买入，否则卖出
        df_lgbm['signal'] = np.where(df_lgbm['predicted_close'] > df_lgbm['close'], 1, -1)
        df_lgbm['signal'] = df_lgbm['signal'].fillna(0)
        
        # 8. 运行回测
        metrics, backtest_results = run_vectorized_backtest(df_lgbm, signal_col='signal')
        
        print(f"✅ LightGBM strategy completed for {ticker}")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        return metrics, backtest_results
        
    except Exception as e:
        print(f"❌ LightGBM strategy failed for {ticker}: {e}")
        
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
    
    # 分割数据
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    try:
        metrics, _ = run_lightgbm_strategy(train_df, test_df, 'TEST')
        print("LightGBM strategy test completed!")
    except Exception as e:
        print(f"LightGBM test failed: {e}")

"""
LSTM 策略
基于LSTM神经网络的时间序列预测交易策略
"""

import pandas as pd
import numpy as np
import warnings
import logging
import os

# 抑制各种模型训练信息
warnings.filterwarnings("ignore", message="ignoring user defined")
warnings.filterwarnings("ignore", message="The model does not support")
warnings.filterwarnings("ignore", message="Failed to use covariates")
warnings.filterwarnings("ignore", message="Training without covariates")

# 抑制PyTorch Lightning日志
os.environ['PYTORCH_LIGHTNING_LOGGING_LEVEL'] = 'ERROR'
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

try:
    from .data_utils import run_vectorized_backtest
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from data_utils import run_vectorized_backtest


def run_lstm_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    """
    LSTM策略 - 基于ETS-SDA原始实现
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        ticker: 股票代码
        
    Returns:
        tuple: (metrics, backtest_results)
    """
    print(f"Running LSTM strategy for {ticker}...")
    
    try:
        # 使用纯PyTorch实现，避免darts库的pandas兼容性问题
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        
        # 1. 数据预处理 - 按照ETS-SDA的方式
        train_df = train_df.copy().dropna()
        test_df = test_df.copy().dropna()

        # 特征选择
        feature_cols = [col for col in train_df.columns if
                        col not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]

        # 确保数据类型正确
        train_df[feature_cols] = train_df[feature_cols].astype('float32')
        test_df[feature_cols] = test_df[feature_cols].astype('float32')
        train_df['close'] = train_df['close'].astype('float32')
        test_df['close'] = test_df['close'].astype('float32')

        # 2. 定义PyTorch LSTM模型
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out
        
        # 3. 数据预处理
        print("   Preprocessing data for PyTorch LSTM...")
        
        # 准备特征数据
        if len(feature_cols) == 0:
            # 如果没有技术指标，使用基本价格特征
            feature_cols = ['close']
        
        # 合并训练和测试数据
        combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        
        # 准备序列数据
        sequence_length = 20  # 使用20天的历史数据预测下一天
        scaler = MinMaxScaler()
        
        # 标准化特征
        features = combined_df[feature_cols].values
        features_scaled = scaler.fit_transform(features)
        
        # 创建序列数据
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i, 0])  # 预测close价格
            return np.array(X), np.array(y)
        
        X, y = create_sequences(features_scaled, sequence_length)
        
        # 分割训练和测试数据
        train_size = len(train_df) - sequence_length
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 4. 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        # 5. 训练LSTM模型
        print("   Training PyTorch LSTM model...")
        input_size = len(feature_cols)
        hidden_size = 64
        num_layers = 2
        output_size = 1
        
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练循环
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
        # 6. 预测
        print("   Making predictions...")
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor).squeeze().numpy()
        
        # 反归一化预测结果 - 只对第一列（close价格）进行反归一化
        # 创建与原始特征相同维度的数组，只填充第一列
        predictions_full = np.zeros((len(predictions), len(feature_cols)))
        predictions_full[:, 0] = predictions
        predictions_rescaled = scaler.inverse_transform(predictions_full)[:, 0]
        
        # 7. 生成交易信号
        df_lstm = test_df.copy()
        
        # 基于预测价格生成信号
        current_prices = test_df['close'].values[sequence_length:]
        predicted_prices = predictions_rescaled[:len(current_prices)]
        
        # 计算预测收益率
        predicted_returns = (predicted_prices - current_prices) / current_prices
        
        # 生成信号
        signals = []
        threshold = 0.01  # 1%的阈值
        
        for ret in predicted_returns:
            if ret > threshold:
                signals.append(1)  # 买入
            elif ret < -threshold:
                signals.append(-1)  # 卖出
            else:
                signals.append(0)  # 持有
        
        # 确保信号长度与测试数据匹配
        if len(signals) < len(test_df):
            signals = [0] * (len(test_df) - len(signals)) + signals
        
        df_lstm['signal'] = signals[:len(test_df)]
        df_lstm['signal'] = df_lstm['signal'].fillna(0)

        # 8. 回测
        metrics, backtest_results = run_vectorized_backtest(df_lstm, signal_col='signal')
        
        print(f"✅ PyTorch LSTM strategy completed for {ticker}")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        # 诊断信息
        signal_counts = pd.Series(signals).value_counts()
        print(f"   Signal distribution: Buy={signal_counts.get(1, 0)}, "
              f"Sell={signal_counts.get(-1, 0)}, Hold={signal_counts.get(0, 0)}")
        
        return metrics, backtest_results
        
    except Exception as e:
        print(f"❌ LSTM strategy failed for {ticker}: {e}")
        # 不使用备用策略，直接返回失败结果
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
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
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
        metrics, _ = run_lstm_strategy(train_df, test_df, 'TEST')
        print("LSTM strategy test completed!")
    except Exception as e:
        print(f"LSTM test failed: {e}")
        print("This is expected if darts/pytorch is not installed properly")

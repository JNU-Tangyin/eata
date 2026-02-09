"""
EATA 策略
调用原始EATA算法进行交易
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings

# 抑制所有数学运算警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all='ignore')

# 抑制matplotlib字体警告
warnings.filterwarnings("ignore", message="findfont: Generic family")
warnings.filterwarnings("ignore", message="Glyph.*missing from current font")

# 添加项目根目录到路径，以便导入真正的EATA本体
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

try:
    from predict import run_eata_core_backtest
    REAL_EATA_AVAILABLE = True
    print("✅ 成功导入 EATA 本体核心回测函数 run_eata_core_backtest")
except ImportError as e:
    print(f"❌ 无法导入 EATA 本体核心函数: {e}")
    REAL_EATA_AVAILABLE = False

# fallback 部分仍然需要向量化回测工具
try:
    from .data_utils import run_vectorized_backtest
except ImportError:
    from data_utils import run_vectorized_backtest


def run_eata_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    """
    EATA策略 - 使用真正的EATA算法
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        ticker: 股票代码
        
    Returns:
        tuple: (metrics, backtest_results)
    """
    print(f"Running EATA strategy for {ticker}...")
    
    if not REAL_EATA_AVAILABLE:
        print("❌ EATA 本体核心不可用，使用备用策略...")
        return _run_fallback_eata_strategy(train_df, test_df, ticker)
    
    try:
        print("   Using EATA core backtest from predict.py (本体回测逻辑)...")

        # 准备 EATA 需要的数据格式：使用 train+test 的完整数据，与本体一致
        combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in combined_df.columns for col in required_cols):
            print(f"   缺少必要列，使用备用策略...")
            return _run_fallback_eata_strategy(train_df, test_df, ticker)

        if 'amount' not in combined_df.columns:
            combined_df['amount'] = combined_df['close'] * combined_df['volume']

        lookback = 50
        lookahead = 10
        stride = 1
        depth = 300
        print(f"   EATA parameters: lookback={lookback}, lookahead={lookahead}, stride={stride}, depth={depth}")

        # 调用本体核心回测函数
        core_metrics, portfolio_df = run_eata_core_backtest(
            stock_df=combined_df,
            ticker=ticker,
            lookback=lookback,
            lookahead=lookahead,
            stride=stride,
            depth=depth,
        )

        # 将本体指标映射到 baseline 期望的格式
        annual_return = core_metrics.get('Annual Return (AR)', 0.0)
        sharpe = core_metrics.get('Sharpe Ratio', 0.0)
        max_dd = core_metrics.get('Max Drawdown (MDD)', 0.0)

        if portfolio_df.empty:
            total_return = 0.0
        else:
            total_return = portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0] - 1.0

        performance_metrics = pd.Series({
            'annualized_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_return': total_return,
        })

        equity_curve = portfolio_df.reset_index().rename(
            columns={'index': 'date', 'value': 'portfolio_value'}
        )

        print(f"✅ Real EATA strategy (core) completed for {ticker}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Annualized Return: {annual_return:.2%}")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Max Drawdown: {max_dd:.2%}")

        return performance_metrics, equity_curve
        
    except Exception as e:
        print(f"❌ EATA strategy simulation failed for {ticker}: {e}")
        
        # 备用策略：基于技术指标的简单规则
        try:
            df_fallback = test_df.copy()
            signals = []
            
            for i in range(len(test_df)):
                current_data = test_df.iloc[i]
                rsi = current_data.get('rsi_14', 50)
                macd = current_data.get('macd_12_26_9', 0)
                macds = current_data.get('macds_12_26_9', 0)
                
                # 简单的多因子规则
                if rsi < 35 and macd > macds:
                    signal = 1  # 买入
                elif rsi > 65 and macd < macds:
                    signal = -1  # 卖出
                else:
                    signal = 0  # 持有
                
                signals.append(signal)
            
            df_fallback['signal'] = signals
            metrics, backtest_results = run_vectorized_backtest(df_fallback, signal_col='signal')
            
            print(f"✅ EATA fallback strategy completed for {ticker}")
            return metrics, backtest_results
            
        except Exception as fallback_error:
            print(f"❌ EATA fallback strategy also failed: {fallback_error}")
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


def _run_fallback_eata_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    """
    EATA备用策略 - 基于技术指标的多因子分析
    """
    print(f"   Using EATA-inspired fallback strategy for {ticker}...")
    
    try:
        signals = []
        
        for i in range(len(test_df)):
            current_data = test_df.iloc[i]
            
            # 技术指标因子
            rsi = current_data.get('rsi_14', 50)
            macd = current_data.get('macd_12_26_9', 0)
            macds = current_data.get('macds_12_26_9', 0)
            
            # 价格动量因子
            if i >= 5:
                short_momentum = (current_data['close'] - test_df.iloc[i-5]['close']) / test_df.iloc[i-5]['close']
            else:
                short_momentum = 0
            
            if i >= 20:
                long_momentum = (current_data['close'] - test_df.iloc[i-20]['close']) / test_df.iloc[i-20]['close']
            else:
                long_momentum = 0
            
            # EATA式的符号回归公式模拟
            score = 0
            
            # RSI因子权重
            if rsi < 30:
                score += 2.0
            elif rsi < 40:
                score += 1.0
            elif rsi > 70:
                score -= 2.0
            elif rsi > 60:
                score -= 1.0
            
            # MACD因子权重
            macd_signal = 1 if macd > macds else -1
            score += macd_signal * 0.8
            
            # 动量因子权重（非线性）
            momentum_score = np.tanh(short_momentum * 10) * 1.5 + np.tanh(long_momentum * 5) * 1.0
            score += momentum_score
            
            # 生成最终信号
            if score >= 1.0:
                signal = 1  # 买入
            elif score <= -1.0:
                signal = -1  # 卖出
            else:
                signal = 0  # 持有
            
            signals.append(signal)
        
        # 创建包含信号的数据框
        test_with_signals = test_df.copy()
        test_with_signals['signal'] = signals
        
        # 运行回测
        metrics, backtest_results = run_vectorized_backtest(test_with_signals, signal_col='signal')
        
        print(f"✅ EATA fallback strategy completed for {ticker}")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        signal_counts = pd.Series(signals).value_counts()
        print(f"   Signal distribution: Buy={signal_counts.get(1, 0)}, "
              f"Sell={signal_counts.get(-1, 0)}, Hold={signal_counts.get(0, 0)}")
        print(f"   Note: Using EATA-inspired technical indicator fallback")
        
        return metrics, backtest_results
        
    except Exception as e:
        print(f"❌ EATA fallback strategy failed: {e}")
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
        metrics, _ = run_eata_strategy(train_df, test_df, 'TEST')
        print("EATA strategy test completed!")
    except Exception as e:
        print(f"EATA test failed: {e}")
        print("This is expected if EATA components are not available")

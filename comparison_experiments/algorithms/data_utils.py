import pandas as pd
import numpy as np
# 使用自定义技术指标计算
# import pandas_ta as ta
import os
from pathlib import Path

# 确保DATA_DIR是相对于项目根目录的路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2020-12-31"
TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2023-12-31"

def load_real_stock_data(ticker: str, db_path: str = "stock.db") -> pd.DataFrame:
    """
    从数据库加载真实股票数据
    
    Args:
        ticker: 股票代码 (e.g., 'AAPL')
        db_path: 数据库文件路径
        
    Returns:
        pd.DataFrame: 包含股票数据的DataFrame
    """
    import sqlite3
    import os
    
    # 构建数据库路径
    if not os.path.isabs(db_path):
        # 相对路径，从项目根目录开始
        project_root = Path(__file__).resolve().parents[2]
        db_path = project_root / db_path
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    # 连接数据库并查询数据
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT date, open, high, low, close, volume, amount 
    FROM downloaded 
    WHERE code = ? 
    ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    # 添加ticker列
    df['ticker'] = ticker
    
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def load_and_preprocess_data(dataset_name: str, ticker: str) -> pd.DataFrame:
    """
    Loads a single stock's data, preprocesses it, and adds technical indicators.

    :param dataset_name: The name of the dataset directory (e.g., 'djia30').
    :param ticker: The stock ticker symbol (e.g., 'AAPL').
    :return: A pandas DataFrame with technical indicators.
    """
    # 优先尝试从数据库加载真实数据
    try:
        df = load_real_stock_data(ticker)
        print(f"✅ 从数据库加载真实股票数据: {ticker}")
    except (FileNotFoundError, ValueError) as e:
        print(f"⚠️  无法从数据库加载 {ticker}: {e}")
        print(f"🔄 尝试从CSV文件加载...")
        
        # 回退到原来的CSV文件加载方式
        file_path = DATA_DIR / dataset_name / f"{ticker}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        df = pd.read_csv(file_path)

    # The finrl library expects date column to be named 'date'
    # and other columns to be in lowercase.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 确保收盘价列存在并统一命名为 'close'
    close_col = None
    for col in df.columns:
        if 'close' in col.lower():  # 匹配 'close' 或 'close_{ticker}'
            close_col = col
            break

    if close_col is None:
        raise ValueError(f"No close price column found for {ticker}")

    # 重命名收盘价列为 'close'
    df = df.rename(columns={close_col: 'close'})
    
    # 验证关键价格列是否存在NaN
    if 'Close' in df.columns and df['Close'].isna().any():
        print(f"WARNING: {ticker} has {df['Close'].isna().sum()} NaN values in Close column")
    
    # 对价格列采用前向填充而非简单填0
    price_cols = ['Open', 'High', 'Low', 'Close']
    price_cols_lower = [col.lower() for col in price_cols]
    for col in price_cols:
        if col in df.columns and df[col].isna().any():
            # 使用前向填充而非填0，保持价格连续性
            df[col] = df[col].fillna(method='ffill')
            # 如果前向填充后仍有NaN（通常是开始处），使用后向填充
            df[col] = df[col].fillna(method='bfill')
            # 如果仍有NaN，说明数据质量有问题，记录警告
            if df[col].isna().any():
                print(f"WARNING: {ticker} still has {df[col].isna().sum()} NaN values in {col} after forward/backward filling")
    
    # 对非价格列使用更安全的填充方式
    df.fillna(0, inplace=True)
    
    df = df.rename(columns={"Date": "date"})
    df.columns = [x.lower().strip() for x in df.columns]

    # Convert date column to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)

    # Add ticker column for multi-stock processing
    df['ticker'] = ticker



    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the dataframe.

    :param df: The input DataFrame with stock data.
    :return: DataFrame with added technical indicators.
    """
    df = df.copy()
    
    # MACD指标
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd_12_26_9'] = ema_12 - ema_26
    df['macds_12_26_9'] = df['macd_12_26_9'].ewm(span=9).mean()
    df['macdh_12_26_9'] = df['macd_12_26_9'] - df['macds_12_26_9']
    
    # RSI指标
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi_6'] = calculate_rsi(df['close'], 6)
    df['rsi_12'] = calculate_rsi(df['close'], 12)
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    # 移动平均线
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    
    # 布林带
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_middle'] = sma_20
    
    # 价格变化率
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    
    # 成交量指标
    if 'volume' in df.columns:
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # 波动率
    df['volatility_20'] = df['close'].rolling(20).std()
    
    # 填充NaN值
    df = df.ffill().bfill().fillna(0)
    
    return df

def get_split_data(df: pd.DataFrame):
    """
    Splits the data into training and testing sets.

    :param df: The full DataFrame.
    :return: A tuple of (train_df, test_df).
    """
    # 确保日期列类型正确
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    train_df = df[(df.date >= TRAIN_START_DATE) & (df.date <= TRAIN_END_DATE)]
    test_df = df[(df.date >= TEST_START_DATE) & (df.date <= TEST_END_DATE)]
    
    # 再次验证分割后的数据集是否有价格列的NaN值
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in train_df.columns:
            nan_count = train_df[col].isna().sum()
            if nan_count > 0:
                print(f"WARNING: Training set has {nan_count} NaN values in {col} column after split")
    
    return train_df, test_df

def run_vectorized_backtest(df, signal_col='signal'):
    """向量化回测函数"""
    signals = df[signal_col].values
    returns = df['close'].pct_change().fillna(0).values
    
    # 计算策略收益
    strategy_returns = signals[:-1] * returns[1:]  # 信号滞后一期
    
    # 计算累积收益
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
    
    # 计算年化收益率
    if len(strategy_returns) > 0:
        # 假设数据是日频，一年252个交易日
        trading_days = len(strategy_returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    else:
        annualized_return = 0
    
    # 计算夏普比率
    if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # 计算最大回撤
    cumulative_curve = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_curve)
    drawdown = (cumulative_curve - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    
    metrics = pd.Series({
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': abs(max_drawdown),
        'num_trades': np.sum(np.abs(np.diff(signals)))
    })
    
    backtest_results = df.copy()
    backtest_results['strategy_return'] = np.concatenate([[0], strategy_returns])
    backtest_results['cumulative_return'] = np.concatenate([[0], cumulative_returns])
    
    return metrics, backtest_results


if __name__ == '__main__':
    # Example usage:
    print("Testing data_utils.py...")
    try:
        # 1. Load data for a sample stock
        aapl_df = load_and_preprocess_data('djia30', 'AAPL')
        print("\nLoaded AAPL data:")
        print(aapl_df.head())

        # 2. Add technical indicators
        aapl_tech_df = add_technical_indicators(aapl_df)
        print("\nAAPL data with technical indicators:")
        print(aapl_tech_df.head())
        print("\nColumns:", aapl_tech_df.columns.tolist())

        # 3. Split data
        train_data, test_data = get_split_data(aapl_tech_df)
        print(f"\nTraining data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
        print("\nScript executed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


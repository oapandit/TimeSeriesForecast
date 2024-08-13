import pandas as pd
import numpy as np

def preprocess_data(file_path, lags=5):
    """
    Preprocess the raw stock price data by creating lagged features and technical indicators.
    
    Args:
        file_path (str): Path to the raw data file.
        lags (int): Number of previous days to use as features.
    
    Returns:
        pd.DataFrame: Processed data with lagged features and technical indicators.
    """
    df = pd.read_csv(file_path)
    
    # Ensure 'Date' is in datetime format and sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Create lagged features
    for i in range(1, lags + 1):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    
    # Add technical indicators
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    
    # Drop rows with NaN values
    df = df.dropna().reset_index(drop=True)
    
    return df

def compute_rsi(series, window):
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    """Compute Moving Average Convergence Divergence (MACD)."""
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

if __name__ == "__main__":
    raw_data_path = 'data/raw/stock_prices.csv'
    processed_data_path = 'data/processed/processed_stock_prices.csv'
    
    processed_data = preprocess_data(raw_data_path, lags=5)
    processed_data.to_csv(processed_data_path, index=False)
    print(f'Processed data saved to {processed_data_path}')



def preprocess_data(raw_data_path):
    raw_data = pd.read_csv(raw_data_path)
    # Example preprocessing: fill missing values, normalize data, etc.
    raw_data.fillna(method='ffill', inplace=True)
    processed_data = raw_data[['Date', 'Close']]
    processed_data.set_index('Date', inplace=True)
    return processed_data


import pandas as pd
import numpy as np

def add_technical_features(df: pd.DataFrame, price_col='Close') -> pd.DataFrame:
    """
    Add common technical indicators to the price dataframe.
    """
    df = df.copy()
    
    # Returns
    df['return'] = df[price_col].pct_change()
    df['vol_10'] = df['return'].rolling(window=10).std()
    
    # Simple Moving Averages
    df['sma_10'] = df[price_col].rolling(window=10).mean()
    df['sma_50'] = df[price_col].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df[price_col].rolling(window=20).mean()
    df['bb_std'] = df[price_col].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
    
    # RSI (Relative Strength Index)
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'].fillna(50)

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df[price_col].shift())
    low_close = np.abs(df['Low'] - df[price_col].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr_14'] = true_range.rolling(14).mean()

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * ((df[price_col] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df[price_col].diff()) * df['Volume']).fillna(0).cumsum()
    
    # --- Lagged Features (Memory) ---
    for lag in [1, 2, 3]:
        df[f'return_lag{lag}'] = df['return'].shift(lag)
        df[f'vol_lag{lag}'] = df['vol_10'].shift(lag)
        df[f'rsi_lag{lag}'] = df['rsi_14'].shift(lag)
        df[f'macd_lag{lag}'] = df['macd'].shift(lag)
    
    df = df.dropna()
    return df

def merge_macro_with_prices(price_df: pd.DataFrame, macro_df: pd.DataFrame, macro_col='value') -> pd.DataFrame:
    """
    Merge macroeconomic data (e.g. GDP, interest rates) with price data on date.
    Assumes both have 'date' columns of datetime type.
    Forward-fill macro data to align with daily price data.
    """
    df = price_df.copy()
    macro_df = macro_df[['date', macro_col]].copy()
    macro_df = macro_df.sort_values('date')
    macro_df = macro_df.set_index('date').resample('D').ffill().reset_index()  # daily frequency

    merged = pd.merge_asof(df.sort_values('date'), macro_df.sort_values('date'), on='date')
    merged = merged.rename(columns={macro_col: 'macro_value'})
    return merged

if __name__ == '__main__':
    # Quick test with yfinance sample data
    import yfinance as yf
    try:
        df = yf.download('AAPL', period='1y')
        if not df.empty:
            df = add_technical_features(df)
            print(df.tail())
    except Exception as e:
        print(f"Test failed: {e}")
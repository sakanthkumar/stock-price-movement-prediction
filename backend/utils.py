import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from dotenv import load_dotenv

# Load env vars if not already loaded
load_dotenv()

ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FRED_API_KEY = os.getenv("FRED_KEY")

# FRED Series IDs
CPI_SERIES_ID = "CPIAUCSL"
GDP_SERIES_ID = "GDP"
UNRATE_SERIES_ID = "UNRATE"

def add_technical_features(df: pd.DataFrame, price_col='Close') -> pd.DataFrame:
    """
    Add common technical indicators to the price dataframe.
    """
    df = df.copy()
    
    # Ensure price_col is a Series (handle duplicate columns)
    if isinstance(df[price_col], pd.DataFrame):
        # If multiple columns with same name, take the first one
        df[price_col] = df[price_col].iloc[:, 0]
        
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
    df['rsi_14'] = df['rsi_14'].fillna(50) # Fill NaN with neutral 50
    
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

    # --- New Quant Indicators ---
    # Bollinger Band Width
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # ROC (Rate of Change) - Momentum
    df['roc_10'] = df[price_col].pct_change(periods=10) * 100

    # ADX (Average Directional Index)
    # TR (True Range) is already calculated for ATR but we need the components for ADX
    # We need to recalculate or reuse components. Let's reuse if possible or recalculate for clarity.
    # +DM and -DM
    up = df['High'].diff()
    down = -df['Low'].diff()
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr = df['atr_14'] # Approximate TR with ATR for smoothing or recalculate exact TR? 
    # Standard ADX uses smoothed TR and smoothed DM.
    # Let's do a simple implementation of ADX 14
    
    # Smooth function
    def smooth(series, window):
        return series.ewm(alpha=1/window, adjust=False).mean()
        
    tr_smooth = smooth(pd.Series(true_range, index=df.index), 14)
    plus_dm_smooth = smooth(pd.Series(plus_dm, index=df.index), 14)
    minus_dm_smooth = smooth(pd.Series(minus_dm, index=df.index), 14)
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx_14'] = smooth(dx, 14)
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # --- Lagged Features (Memory) ---
    for lag in [1, 2, 3]:
        df[f'return_lag{lag}'] = df['return'].shift(lag)
        df[f'vol_lag{lag}'] = df['vol_10'].shift(lag)
        df[f'rsi_lag{lag}'] = df['rsi_14'].shift(lag)
        df[f'macd_lag{lag}'] = df['macd'].shift(lag)

    # Drop NaNs created by rolling windows
    # We don't drop here because we might need the last row even if some history is missing, 
    # but for prediction we need valid features.
    # For live prediction, we fetch enough history (e.g. 60 days) so the last row is valid.
    
    # Fill NaNs to prevent model crashes
    # Forward fill first (propagate last valid observation forward)
    df = df.ffill()
    # Backward fill (propagate next valid observation backward)
    df = df.bfill()
    # If still NaNs (e.g. empty or all NaN), fill with 0
    df = df.fillna(0)
    
    # Add Candlestick Patterns
    df = add_candlestick_features(df)
    
    return df

def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add manual candlestick pattern detection.
    Patterns: Doji, Hammer, Bullish/Bearish Engulfing.
    """
    df = df.copy()
    
    # Basic Candle Properties
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    
    # 1. Doji: Body is very small relative to Range
    # Condition: Body <= 10% of Range
    df['Doji'] = np.where(df['Body'] <= 0.1 * df['Range'], 1, 0)
    
    # 2. Hammer: Small body, long lower shadow (>= 2x body), small upper shadow
    # Condition: Lower Shadow >= 2 * Body AND Upper Shadow <= Body
    df['Hammer'] = np.where(
        (df['Lower_Shadow'] >= 2 * df['Body']) & 
        (df['Upper_Shadow'] <= df['Body']), 
        1, 0
    )
    
    # 3. Engulfing Patterns (Requires previous candle)
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Body'] = abs(df['Prev_Close'] - df['Prev_Open'])
    
    # Bullish Engulfing: 
    # Current is Green (Close > Open), Previous was Red (Prev_Close < Prev_Open)
    # Current Body engulfs Previous Body (Open < Prev_Close AND Close > Prev_Open)
    df['Bullish_Engulfing'] = np.where(
        (df['Close'] > df['Open']) & 
        (df['Prev_Close'] < df['Prev_Open']) & 
        (df['Open'] < df['Prev_Close']) & 
        (df['Close'] > df['Prev_Open']),
        1, 0
    )
    
    # Bearish Engulfing:
    # Current is Red (Close < Open), Previous was Green (Prev_Close > Prev_Open)
    # Current Body engulfs Previous Body (Open > Prev_Close AND Close < Prev_Open)
    df['Bearish_Engulfing'] = np.where(
        (df['Close'] < df['Open']) & 
        (df['Prev_Close'] > df['Prev_Open']) & 
        (df['Open'] > df['Prev_Close']) & 
        (df['Close'] < df['Prev_Open']),
        1, 0
    )
    
    # Cleanup temporary columns
    df.drop(columns=['Body', 'Range', 'Upper_Shadow', 'Lower_Shadow', 
                     'Prev_Open', 'Prev_Close', 'Prev_Body'], inplace=True, errors='ignore')
                     
    return df

def fetch_price_data(ticker: str, period="3mo", interval="1d"):
    """
    Fetch price data from yfinance with retries.
    Tries yf.download first, then yf.Ticker.history as fallback.
    """
    import time
    
    # Clean ticker
    ticker = ticker.strip().upper()
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Method 1: yf.download (Bulk optimized, but sometimes fails for single)
            # Add delay between retries
            if attempt > 0: time.sleep(1)
                
            df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=20, auto_adjust=True)
            
            # Check if valid
            if df is not None and not df.empty:
                # Handle MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        df.columns = df.columns.get_level_values(0)
                    except IndexError: pass
                
                df = df.reset_index()
                df.columns = [c.capitalize() for c in df.columns]
                df = df.loc[:, ~df.columns.duplicated()]
                
                req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in req_cols):
                    return df

            # Method 2: yf.Ticker.history (Fallback)
            print(f"⚠️ Attempt {attempt+1}: yf.download failed/empty for {ticker}, trying Ticker.history...")
            dat = yf.Ticker(ticker)
            df = dat.history(period=period, interval=interval, auto_adjust=True)
            
            if df is not None and not df.empty:
                df = df.reset_index()
                df.columns = [c.capitalize() for c in df.columns]
                df = df.loc[:, ~df.columns.duplicated()]
                
                # Rename 'Stock splits' etc if needed, but we just need OHLCV
                req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in req_cols):
                    return df
            
            print(f"⚠️ Attempt {attempt+1}: Both methods failed for {ticker}")

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}: Error fetching data for {ticker}: {e}")
            
    print(f"❌ Failed to fetch data for {ticker} after {max_retries} attempts.")
    return None

def fetch_fred_data(series_id: str, api_key: str):
    if not api_key:
        return 0.0
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
        r = requests.get(url, params=params, timeout=30) # Increased timeout
        obs = r.json().get("observations", [])
        if not obs:
            return 0.0
        latest = obs[-1]
        return float(latest["value"]) if latest["value"] != "." else 0.0
    except Exception as e:
        print(f"FRED fetch failed for {series_id}: {e}")
        return 0.0

def get_macro_data():
    """
    Fetch current macro data.
    """
    cpi = fetch_fred_data(CPI_SERIES_ID, FRED_API_KEY)
    gdp = fetch_fred_data(GDP_SERIES_ID, FRED_API_KEY)
    unrate = fetch_fred_data(UNRATE_SERIES_ID, FRED_API_KEY)
    return {"CPI": cpi, "GDP": gdp, "UNRATE": unrate}

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment(ticker: str):
    """
    Fetch recent news for the ticker using NewsAPI and calculate sentiment score.
    Returns a score between 0 and 1 (0=Negative, 0.5=Neutral, 1=Positive).
    """
    if not NEWSAPI_KEY:
        return 0.5
        
    try:
        # Clean ticker for news search (e.g. "BTC-USD" -> "Bitcoin", "AAPL" -> "Apple")
        query = ticker
        if "-USD" in ticker:
            query = ticker.split("-")[0]
            if query == "BTC": query = "Bitcoin"
            if query == "ETH": query = "Ethereum"
            
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": NEWSAPI_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10  # Analyze last 10 articles
        }
        
        response = requests.get(url, params=params, timeout=30) # Increased timeout
        data = response.json()
        
        articles = data.get("articles", [])
        if not articles:
            return 0.5
            
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title} {description}"
            
            # VADER returns compound score -1 to 1
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)
            
        if not scores:
            return 0.5
            
        avg_score = sum(scores) / len(scores)
        
        # Normalize from [-1, 1] to [0, 1] for the model
        # -1 -> 0, 0 -> 0.5, 1 -> 1
        normalized_score = (avg_score + 1) / 2
        return normalized_score
        
    except Exception as e:
        print(f"⚠️ Error fetching sentiment for {ticker}: {e}")
        return 0.5

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("⚠️ CCXT not found. Falling back to yfinance for crypto prices.")

def get_live_price(ticker: str):
    """
    Fetch current live price.
    Uses CCXT for Crypto (Real-time) and yfinance for Stocks.
    """
    try:
        # 1. Try CCXT for Crypto (BTC-USD, ETH-USD)
        if HAS_CCXT and ("-USD" in ticker or ticker in ["BTC", "ETH", "DOGE", "SOL"]):
            symbol = ticker.replace("-USD", "/USDT") # Binance uses USDT
            if ticker == "BTC": symbol = "BTC/USDT"
            if ticker == "ETH": symbol = "ETH/USDT"
            
            try:
                exchange = ccxt.binanceus() # Use Binance US for USD pairs
                ticker_data = exchange.fetch_ticker(symbol)
                return float(ticker_data['last'])
            except Exception:
                # Fallback to Coinbase if Binance fails
                try:
                    exchange = ccxt.coinbase()
                    symbol = ticker.replace("-USD", "/USD")
                    ticker_data = exchange.fetch_ticker(symbol)
                    return float(ticker_data['last'])
                except Exception as e:
                    print(f"CCXT Error for {ticker}: {e}")
        
        # 2. yfinance for Stocks (or crypto fallback)
        stock = yf.Ticker(ticker)
        
        # Method 1: fast_info (Newer, faster, less prone to 401)
        try:
            price = stock.fast_info.last_price
            if price: return float(price)
        except Exception:
            pass

        # Method 2: 1m history (Reliable for active market)
        try:
            # period="1d" gets today's data. If market just opened, might be empty if no trades yet?
            # "1d" usually returns at least the last minute if open.
            df = stock.history(period="1d", interval="1m")
            if not df.empty:
                return float(df.iloc[-1]['Close'])
        except Exception:
            pass

        # Method 3: info dictionary (Legacy, slow, prone to 401)
        try:
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('ask')
            if price: return float(price)
        except Exception:
            pass
            
        return None
    except Exception as e:
        print(f"Error fetching live price for {ticker}: {e}")
        return None

def is_market_open(ticker: str) -> bool:
    """
    Check if market is open for the given ticker.
    Crypto (BTC, ETH) is always open.
    For stocks, we'll assume Open if the user says so, or check if we can get a live price update.
    """
    if "-USD" in ticker or ticker in ["BTC", "ETH", "DOGE"]:
        return True
        
    # US Market Hours (Strict)
    from datetime import datetime, time
    import pytz
    
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    # Check Weekend
    if now.weekday() >= 5: # Sat=5, Sun=6
        return False
        
    # Check Time (9:30 - 16:00)
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= now.time() <= market_close

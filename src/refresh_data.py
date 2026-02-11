import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from data_collection import fetch_price_history_yfinance, fetch_fred_series
# from preprocess import add_technical_features # Not needed here, done in training

# Configuration
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "INTC", "NFLX", # Tech
    "JPM", "BAC", "V", "MA", "GS", # Finance
    "WMT", "PG", "KO", "DIS", "MCD", # Consumer
    "SPY", "QQQ", # Indices
    "BTC-USD", "ETH-USD" # Crypto
]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(DATA_DIR, "all_tickers_with_macro_sentiment.csv")
INTRADAY_DIR = os.path.join(DATA_DIR, "intraday")

def fetch_intraday_data(ticker):
    """
    Fetch 1-minute intraday data for the last 7 days.
    """
    try:
        # 1m data is available for last 7 days
        df = yf.download(ticker, period="7d", interval="1m", progress=False)
        if df.empty:
            print(f"Warning: No intraday data for {ticker}")
            return None
        
        # yfinance returns multi-index columns if multiple tickers, but here we do one by one.
        # It might return columns like (Price, Ticker). We need to flatten if necessary.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df['Ticker'] = ticker
        df.reset_index(inplace=True)
        # Rename Datetime to Date for consistency
        df = df.rename(columns={'Datetime': 'Date', 'index': 'Date'})
        return df
    except Exception as e:
        print(f"Error fetching intraday data for {ticker}: {e}")
        return None

def main():
    print("Starting data refresh...")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INTRADAY_DIR, exist_ok=True)
    
    # --- 1. Daily Data ---
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d') # 5 years
    
    print(f"Fetching Daily data from {start_date} to {end_date}")
    
    # Fetch Macro Data
    print("Fetching Macro Data...")
    try:
        gdp_df = fetch_fred_series('GDP', start_date=start_date)
        unrate_df = fetch_fred_series('UNRATE', start_date=start_date)
        cpi_df = fetch_fred_series('CPIAUCSL', start_date=start_date)
    except Exception as e:
        print(f"Macro fetch failed: {e}. Using placeholders.")
        gdp_df = pd.DataFrame()
        unrate_df = pd.DataFrame()
        cpi_df = pd.DataFrame()

    all_daily_data = []

    for ticker in TICKERS:
        print(f"Processing {ticker} (Daily)...")
        try:
            # Price Data
            df = fetch_price_history_yfinance(ticker, start=start_date, end=end_date)
            if df.empty:
                print(f"No daily data for {ticker}")
                continue
                
            df['Ticker'] = ticker
            
            # Merge Macro (ffill)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            elif 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # Helper to merge macro
            def merge_macro(source_df, col_name):
                if source_df.empty:
                    df[col_name] = 0.0
                    return
                source_df = source_df.copy()
                source_df['date'] = pd.to_datetime(source_df['date']).dt.tz_localize(None)
                source_df = source_df.sort_values('date')
                merged = pd.merge_asof(df.sort_values('date'), source_df[['date', 'value']], on='date', direction='backward')
                df[col_name] = merged['value']

            merge_macro(gdp_df, 'GDP')
            merge_macro(unrate_df, 'UNRATE')
            merge_macro(cpi_df, 'CPI')
            
            df['sentiment'] = 0.5 # Placeholder
            
            # Rename date to Date
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'Date'})
            
            all_daily_data.append(df)
            
        except Exception as e:
            print(f"Failed Daily {ticker}: {e}")

    if all_daily_data:
        final_df = pd.concat(all_daily_data)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved Daily data to {OUTPUT_FILE}")
        print(f"Total Daily rows: {len(final_df)}")
    else:
        print("No Daily data collected!")

    # --- 2. Intraday Data ---
    print(f"\nFetching Intraday (1m) data...")
    
    for ticker in TICKERS:
        print(f"Processing {ticker} (1m)...")
        df_intra = fetch_intraday_data(ticker)
        if df_intra is not None:
            output_path = os.path.join(INTRADAY_DIR, f"{ticker}_1m.csv")
            df_intra.to_csv(output_path, index=False)
            print(f"Saved {output_path} ({len(df_intra)} rows)")

    print("\nData refresh complete!")

if __name__ == "__main__":
    main()

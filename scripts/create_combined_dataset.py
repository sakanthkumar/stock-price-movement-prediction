import os
import pandas as pd
import yfinance as yf
import praw
import requests
import logging
from dotenv import load_dotenv
from datetime import datetime

# Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"
FRED_KEY = os.getenv("FRED_KEY")

# === Macro Indicators (FRED) ===
FRED_SERIES = {
    "GDP": "GDP",
    "UNRATE": "UNRATE",
    "CPI": "CPIAUCSL"
}

# Initialize Reddit client
try:
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )
except Exception:
    reddit = None
    logging.warning("Reddit client not initialized, sentiment will be zero.")


def fetch_price_data(ticker):
    """Download stock price history"""
    logging.info(f"Fetching price data for {ticker}")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)

    # Reset index & flatten any multi-index
    df = df.reset_index()
    df.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in df.columns]

    # Keep required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} for {ticker}")

    df = df[required_cols].copy()
    df['Ticker'] = ticker
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def fetch_macro_data():
    """Fetch macroeconomic indicators from FRED"""
    macro_frames = []
    for name, series_id in FRED_SERIES.items():
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_KEY}&file_type=json"
        try:
            logging.info(f"Fetching macro data for {name}")
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()["observations"]
            df = pd.DataFrame(data)[["date", "value"]]
            df.rename(columns={"value": name}, inplace=True)
            df[name] = pd.to_numeric(df[name], errors="coerce")
            df["Date"] = pd.to_datetime(df["date"])
            df.drop(columns=["date"], inplace=True)
            macro_frames.append(df)
        except Exception as e:
            logging.error(f"Error fetching {name}: {e}")

    if not macro_frames:
        return pd.DataFrame()

    # Merge all macro indicators
    macro_df = macro_frames[0]
    for df in macro_frames[1:]:
        macro_df = pd.merge(macro_df, df, on="Date", how="outer")

    # Ensure Date is datetime
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    return macro_df


def get_reddit_sentiment(ticker, limit=50):
    """Simple sentiment scoring from Reddit posts"""
    if reddit is None:
        return 0.0
    try:
        subreddit = reddit.subreddit("stocks")
        posts = subreddit.search(ticker, limit=limit)
        positive_words = ["buy", "bull", "long", "moon", "gain"]
        negative_words = ["sell", "bear", "short", "drop", "loss"]
        score, count = 0, 0
        for post in posts:
            text = (post.title + " " + post.selftext).lower()
            for w in positive_words:
                if w in text:
                    score += 1
            for w in negative_words:
                if w in text:
                    score -= 1
            count += 1
        return score / count if count > 0 else 0.0
    except Exception as e:
        logging.warning(f"Error fetching Reddit sentiment for {ticker}: {e}")
        return 0.0


def build_sentiment_df(tickers):
    logging.info("Building sentiment data for all tickers (once per ticker)...")
    return pd.DataFrame([{"Ticker": t, "sentiment": get_reddit_sentiment(t)} for t in tickers])


def main():
    # === Stock price data ===
    all_price_dfs = [fetch_price_data(t) for t in TICKERS]
    combined_prices = pd.concat(all_price_dfs, ignore_index=True)
    combined_prices['Date'] = pd.to_datetime(combined_prices['Date'])

    # === Macro data ===
    macro_df = fetch_macro_data()
    if not macro_df.empty:
        combined_prices = pd.merge(combined_prices, macro_df, on="Date", how="left")
        for col in macro_df.columns:
            if col != "Date":
                combined_prices[col] = combined_prices[col].ffill().fillna(0)

    # === Sentiment data ===
    sentiment_df = build_sentiment_df(TICKERS)
    combined_prices = combined_prices.merge(sentiment_df, on="Ticker", how="left")
    combined_prices['sentiment'] = combined_prices['sentiment'].fillna(0)

    # === Target variable ===
    combined_prices = combined_prices.sort_values(['Ticker', 'Date'])
    combined_prices['target'] = combined_prices.groupby('Ticker')['Close'].shift(-1)
    combined_prices = combined_prices.dropna(
        subset=['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment', 'target']
    )

    # === Save final dataset ===
    out_path = os.path.join(BASE_DIR, "data", "processed", "all_tickers_with_macro_sentiment.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined_prices.to_csv(out_path, index=False)
    logging.info(f"✅ Combined dataset saved to {out_path}")

    # Preview
    print(combined_prices.head())


# Run script
if __name__ == "__main__":
    main()

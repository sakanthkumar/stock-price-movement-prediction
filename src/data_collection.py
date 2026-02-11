import os
import yfinance as yf
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

FRED_KEY = os.getenv('FRED_KEY')
ALPHAVANTAGE_KEY = os.getenv('ALPHAVANTAGE_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

_vader = SentimentIntensityAnalyzer()

def fetch_price_history_yfinance(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """Fetch stock prices from Yahoo Finance."""
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start=start, end=end, auto_adjust=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    df.rename(columns={'Date': 'date'}, inplace=True)
    return df

def fetch_price_history_alpha_vantage(ticker: str) -> pd.DataFrame:
    """Fetch stock prices from Alpha Vantage as fallback."""
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': ticker,
        'outputsize': 'full',
        'apikey': ALPHAVANTAGE_KEY,
        'datatype': 'json',
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    ts = data.get('Time Series (Daily)', {})
    records = []
    for date, vals in ts.items():
        records.append({
            'date': date,
            'Open': float(vals['1. open']),
            'High': float(vals['2. high']),
            'Low': float(vals['3. low']),
            'Close': float(vals['4. close']),
            'Volume': int(vals['6. volume'])
        })
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def fetch_news(query: str, from_date: str = None, to_date: str = None, page=1):
    """Fetch news articles using NewsAPI (optional, not used for sentiment here)."""
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'pageSize': 100,
        'page': page,
        'apiKey': NEWSAPI_KEY,
        'language': 'en',
        'sortBy': 'relevancy'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_fred_series(series_id: str, start_date: str = None, end_date: str = None):
    """Fetch macroeconomic data series from FRED API."""
    url = f'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': FRED_KEY,
        'file_type': 'json',
    }
    if start_date:
        params['observation_start'] = start_date
    if end_date:
        params['observation_end'] = end_date

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    observations = data.get('observations', [])
    df = pd.DataFrame(observations)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
    return df

def fetch_reddit_sentiment(ticker: str, limit=200) -> pd.DataFrame:
    """
    Fetch recent Reddit submissions mentioning the ticker symbol
    and calculate average sentiment score per day.
    """
    posts = reddit.subreddit('all').search(ticker, limit=limit, sort='new')
    sentiments = {}

    for post in posts:
        date = pd.to_datetime(post.created_utc, unit='s').date()
        text = post.title + ' ' + (post.selftext or '')
        score = _vader.polarity_scores(text)['compound']

        if date in sentiments:
            sentiments[date].append(score)
        else:
            sentiments[date] = [score]

    daily_sentiments = [
        {'date': pd.to_datetime(date), 'sentiment': sum(scores)/len(scores)}
        for date, scores in sentiments.items()
    ]

    return pd.DataFrame(daily_sentiments)

def fetch_daily_sentiment(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Fetching Reddit sentiment for {ticker} from {start_date} to {end_date}...")
    df = fetch_reddit_sentiment(ticker, limit=500)
    # Filter by date range
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    return df

if __name__ == '__main__':
    print(f"Loaded FRED_KEY: {FRED_KEY}")

    print("Fetching AAPL prices (Yahoo Finance)...")
    df_prices = fetch_price_history_yfinance('AAPL', start='2023-01-01', end='2023-06-01')
    print(df_prices.tail())

    print("Fetching AAPL prices (Alpha Vantage)...")
    df_av = fetch_price_history_alpha_vantage('AAPL')
    print(df_av.tail())

    print("Fetching FRED data for GDP (GDP)...")
    fred_df = fetch_fred_series('GDP', start_date='2020-01-01')
    print(fred_df.tail())

    print("Fetching Reddit sentiment for AAPL...")
    sentiment_df = fetch_daily_sentiment('AAPL', '2023-01-01', '2023-06-01')
    print(sentiment_df.head())

    print("Data collection script executed successfully.")
else:
    print("This script is intended to be run as a standalone module.")

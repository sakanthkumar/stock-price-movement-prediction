import os
import pandas as pd
from src.data_collection import (
    fetch_price_history_yfinance,
    fetch_fred_series,
    fetch_daily_sentiment,
)

def add_technical_indicators(df):
    # Placeholder for your technical indicators (e.g., moving averages)
    # For now, return the dataframe as is
    return df

def merge_macro_with_prices(price_df, macro_df):
    # Remove timezone info from price_df date column
    if pd.api.types.is_datetime64tz_dtype(price_df['date']):
        price_df['date'] = price_df['date'].dt.tz_localize(None)
    # Remove timezone info from macro_df date column if exists and needed
    if 'date' in macro_df.columns and pd.api.types.is_datetime64tz_dtype(macro_df['date']):
        macro_df['date'] = macro_df['date'].dt.tz_localize(None)

    merged = pd.merge(price_df, macro_df[['date', 'value']], on='date', how='left')
    merged.rename(columns={'value': 'macro_value'}, inplace=True)
    return merged

def main():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    print("Fetching AAPL price data...")
    price_df = fetch_price_history_yfinance(ticker, start=start_date, end=end_date)

    print("Fetching UNRATE macroeconomic data...")
    macro_df = fetch_fred_series('UNRATE', start_date=start_date, end_date=end_date)

    print("Adding technical indicators...")
    price_features_df = add_technical_indicators(price_df)

    print("Merging macro data with price features...")
    combined_df = merge_macro_with_prices(price_features_df, macro_df)

    print("Fetching Reddit sentiment data...")
    sentiment_df = fetch_daily_sentiment(ticker, start_date, end_date)

    print("Merging sentiment with combined data...")
    final_df = pd.merge(combined_df, sentiment_df, on='date', how='left')

    output_path = f"data/processed/{ticker}_with_macro_sentiment.csv"
    print(f"Saving final data to {output_path} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()

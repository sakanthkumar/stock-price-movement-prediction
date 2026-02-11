import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from preprocess import add_technical_features
import glob

# Set directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
INTRADAY_DIR = os.path.join(DATA_DIR, "intraday")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_save_model(df, model_name, target_shift=-1):
    print(f"\nTraining {model_name}...")
    
    # Process each ticker separately
    tickers = df['Ticker'].unique()
    processed_dfs = []

    print(f"Processing {len(tickers)} tickers...")
    for ticker in tickers:
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date')
        
        # Add technical features
        # Note: For intraday (5m), standard windows (14, 50) might be too short/long depending on strategy.
        # But for consistency and "short term trends", we keep the same feature set.
        ticker_df = add_technical_features(ticker_df, price_col='Close')
        
        # Target: Next period Percentage Change (Return)
        # (Next Close - Current Close) / Current Close
        ticker_df['target'] = (ticker_df['Close'].shift(target_shift) - ticker_df['Close']) / ticker_df['Close']
        
        processed_dfs.append(ticker_df)

    # Combine back
    df_processed = pd.concat(processed_dfs)

    # Drop missing rows
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        # 'GDP', 'UNRATE', 'CPI', 'sentiment', # Macro might be missing in intraday or irrelevant
        'vol_10', 'sma_10', 'sma_50', 'ema_12', 'ema_26', 
        'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'rsi_14',
        'atr_14', 'stoch_k', 'stoch_d', 'obv',
        # Lags
        'return_lag1', 'return_lag2', 'return_lag3',
        'vol_lag1', 'vol_lag2', 'vol_lag3',
        'rsi_lag1', 'rsi_lag2', 'rsi_lag3',
        'macd_lag1', 'macd_lag2', 'macd_lag3'
    ]
    
    # Filter features that actually exist in the dataframe (Intraday might miss macro)
    available_features = [col for col in feature_cols if col in df_processed.columns]
    
    df_processed = df_processed.dropna(subset=available_features + ['target'])
    print(f"Total rows after cleaning: {len(df_processed)}")

    # Features and target
    X = df_processed[available_features]
    y = df_processed['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Train model - XGBoost
    print(f"Training {model_name} with XGBoost...")
    
    if "intraday" in model_name:
        # Intraday: Fast but deep
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    else:
        # Daily: More robust
        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1
        )
        
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")

    # Save
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{model_name}.pkl")
    features_path = os.path.join(MODELS_DIR, f"features_{model_name}.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(available_features, features_path)
    
    print(f"Saved {model_name} to {model_path}")

def main():
    # 1. Train Daily Model
    daily_csv = os.path.join(DATA_DIR, "all_tickers_with_macro_sentiment.csv")
    if os.path.exists(daily_csv):
        print("Loading Daily Data...")
        df_daily = pd.read_csv(daily_csv, parse_dates=['Date'])
        train_and_save_model(df_daily, "model_daily", target_shift=-1)
    else:
        print("Daily data not found!")

    # 2. Train Intraday Model
    print("\nLoading Intraday Data...")
    intraday_files = glob.glob(os.path.join(INTRADAY_DIR, "*_1m.csv"))
    if intraday_files:
        dfs = []
        for f in intraday_files:
            try:
                df = pd.read_csv(f, parse_dates=['Date'])
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if dfs:
            df_intraday = pd.concat(dfs)
            # Intraday data might not have macro columns, which is fine.
            train_and_save_model(df_intraday, "model_intraday", target_shift=-1) # Next 1m candle
        else:
            print("No valid intraday data found.")
    else:
        print("No intraday files found!")

if __name__ == "__main__":
    main()

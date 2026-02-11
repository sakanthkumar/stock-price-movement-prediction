import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess import add_technical_features

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_scalper_model():
    print("Training Scalper Model (XGBClassifier)...")
    
    # We need 5m data. Since we might not have a huge CSV of 5m data saved, 
    # we'll try to use what we have or fetch fresh if needed.
    # For this script, let's assume we can fetch fresh data or use a sample.
    # Ideally, we should have saved 5m data. 
    # Let's check if 'all_tickers_5m.csv' exists, if not, we might need to fetch.
    
    # fallback: fetch for a few volatile tickers to train
    import yfinance as yf
    tickers = ["BTC-USD", "ETH-USD", "NVDA", "TSLA", "AAPL", "AMD"]
    all_data = []
    
    print("Fetching 5m data for training...")
    for ticker in tickers:
        try:
            # Fetch last 7 days (max for 1m)
            df = yf.download(ticker, period="7d", interval="1m", progress=False)
            if df.empty: continue
            
            # Fix MultiIndex immediately
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df['Ticker'] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            
    if not all_data:
        print("❌ No data fetched. Cannot train.")
        return

    df = pd.concat(all_data)
    df = df.reset_index() # Ensure Date/Datetime is a column
    
    # Rename columns if multi-index (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten: 'Close' level 0, 'Ticker' level 1
        # We want to keep the data but we concatenated multiple tickers.
        # Actually, when we concat list of DFs where each DF has single ticker but might be multi-index...
        # Let's fix each DF BEFORE concat.
        pass
        
    # Better approach: Fix inside the loop
    pass

    # Preprocess
    # We need to handle the column names carefully as yfinance might return 'Datetime'
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})
    elif 'index' in df.columns: # reset_index might create 'index' if name was None
        df = df.rename(columns={'index': 'Date'})
        
    # Add features
    # We need to group by Ticker if we have multiple
    # But add_technical_features expects single DF. 
    # Let's apply to each ticker group
    processed_dfs = []
    for ticker in df['Ticker'].unique():
        sub_df = df[df['Ticker'] == ticker].copy()
        sub_df = sub_df.sort_values('Date')
        sub_df = add_technical_features(sub_df)
        processed_dfs.append(sub_df)
        
    df = pd.concat(processed_dfs)
    
    # Target: 1 if Next Close > Current Close (UP), else 0 (DOWN)
    # Scalping is about immediate next move.
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop last row (NaN target)
    df = df.dropna()
    
    # Features
    feature_cols = [
        'vol_10', 'rsi_14', 'macd', 'macd_signal', 
        'atr_14', 'stoch_k', 'stoch_d', 'obv',
        'return_lag1', 'return_lag2', 'return_lag3',
        'vol_lag1', 'rsi_lag1', 'macd_lag1'
    ]
    
    # Ensure cols exist
    available_feats = [c for c in feature_cols if c in df.columns]
    
    X = df[available_feats]
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train Classifier
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nScalper Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    # Save
    joblib.dump(model, os.path.join(MODELS_DIR, "model_scalper.pkl"))
    joblib.dump(available_feats, os.path.join(MODELS_DIR, "features_model_scalper.pkl"))
    # No scaler needed for Tree models usually, but if we used one before we should be consistent.
    # XGBoost handles unscaled fine. We won't use scaler for this one to keep it fast.
    
    print("✅ Scalper Model Saved.")

if __name__ == "__main__":
    train_scalper_model()

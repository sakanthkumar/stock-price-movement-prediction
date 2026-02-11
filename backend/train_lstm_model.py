import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import utils

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "lstm_daily.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_lstm.pkl")

# Tickers to train on (diverse set)
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "INTC", # Tech
    "JPM", "BAC", "GS", # Finance
    "XOM", "CVX", # Energy
    "PFE", "JNJ", # Healthcare
    "BTC-USD", "ETH-USD" # Crypto
]

SEQ_LENGTH = 60 # Input sequence length (days)
PREDICT_DAYS = 3 # Forecast horizon

def create_sequences(data, seq_length, predict_days):
    X, y = [], []
    for i in range(len(data) - seq_length - predict_days):
        X.append(data[i:(i + seq_length)])
        # Target: Return over next 'predict_days'
        # We want to predict the % change from current close to close 'predict_days' later
        current_close = data[i + seq_length - 1, 3] # Assuming Close is index 3
        future_close = data[i + seq_length + predict_days - 1, 3]
        
        ret = (future_close - current_close) / current_close
        y.append(ret)
        
    return np.array(X), np.array(y)

def train_lstm_model():
    print("🚀 Starting LSTM Model Training...")
    
    all_data = []
    
    # 1. Fetch and Preprocess Data
    for ticker in TICKERS:
        print(f"Fetching data for {ticker}...")
        df = utils.fetch_price_data(ticker, period="2y") # Need enough history
        if df is None or df.empty:
            continue
            
        df = utils.add_technical_features(df)
        
        # Select Features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'rsi_14', 'macd', 'vol_10', 'sma_50', 'ema_12', 
                    'Doji', 'Hammer', 'Bullish_Engulfing', 'Bearish_Engulfing']
        
        # Ensure all features exist
        for col in features:
            if col not in df.columns: df[col] = 0.0
            
        data = df[features].values
        all_data.append(data)

    if not all_data:
        print("❌ No data fetched. Exiting.")
        return

    # Concatenate all data for scaling
    combined_data = np.vstack(all_data)
    
    # Handle NaNs and Infs
    combined_data = np.nan_to_num(combined_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(combined_data)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved to {SCALER_PATH}")
    
    # Create Sequences
    X_train, y_train = [], []
    
    for data in all_data:
        scaled_data = scaler.transform(data)
        X, y = create_sequences(scaled_data, SEQ_LENGTH, PREDICT_DAYS)
        if len(X) > 0:
            X_train.append(X)
            y_train.append(y)
            
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    print(f"Training Data Shape: X={X_train.shape}, y={y_train.shape}")
    
    # 2. Build LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1) # Regression output: Predicted Return
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 3. Train
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
    
    # 4. Save
    model.save(MODEL_PATH)
    print(f"✅ LSTM Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_lstm_model()

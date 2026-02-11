import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from preprocess import add_technical_features

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Config
LOOKBACK = 60 # Days of history to look at
EPOCHS = 10
BATCH_SIZE = 64

def create_sequences(data, target, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def train_lstm_model(df, model_name):
    print(f"\nTraining LSTM: {model_name}...")
    
    # Preprocess
    df = add_technical_features(df)
    
    # Features
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'vol_10', 'sma_10', 'sma_50', 'ema_12', 'ema_26', 
        'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'rsi_14',
        'atr_14', 'stoch_k', 'stoch_d', 'obv',
        'return_lag1', 'return_lag2', 'return_lag3',
        'vol_lag1', 'vol_lag2', 'vol_lag3',
        'rsi_lag1', 'rsi_lag2', 'rsi_lag3',
        'macd_lag1', 'macd_lag2', 'macd_lag3'
    ]
    
    # Filter available
    available_features = [c for c in feature_cols if c in df.columns]
    
    # Target: Next Close (Regression) or Return?
    # Let's predict Return to be consistent with XGBoost
    df['target'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
    df = df.dropna(subset=available_features + ['target'])
    
    if df.empty:
        print("No data.")
        return

    # Scale Features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[available_features])
    target_scaled = df['target'].values # Target doesn't strictly need scaling if it's small returns, but safer to scale
    
    # Create Sequences
    X, y = create_sequences(data_scaled, target_scaled, LOOKBACK)
    
    if len(X) < 100:
        print("Not enough data for LSTM.")
        return

    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Build Model
    model = Sequential()
    # Input shape: (lookback, n_features)
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Predict 1 value (return)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)
    
    # Save
    model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{model_name}_lstm.pkl")
    features_path = os.path.join(MODELS_DIR, f"features_{model_name}_lstm.pkl")
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(available_features, features_path)
    
    print(f"Saved LSTM to {model_path}")

def main():
    # Train Daily
    daily_csv = os.path.join(DATA_DIR, "all_tickers_with_macro_sentiment.csv")
    if os.path.exists(daily_csv):
        df = pd.read_csv(daily_csv, parse_dates=['Date'])
        train_lstm_model(df, "lstm_daily")
        
    # Train Intraday (Optional, might be too slow/heavy for now)
    # Let's focus on Daily first as it's the "Nuclear Option" for swing trading
    
if __name__ == "__main__":
    main()

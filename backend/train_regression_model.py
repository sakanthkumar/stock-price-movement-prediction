import os
import sys
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Add backend to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import utils
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'backend'))
    import utils

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Artifact Paths
MODEL_DAILY_PATH = os.path.join(MODELS_DIR, "model_daily.pkl")
SCALER_DAILY_PATH = os.path.join(MODELS_DIR, "scaler_model_daily.pkl")
FEATURES_DAILY_PATH = os.path.join(MODELS_DIR, "features_model_daily.pkl")

# Metrics file
METRICS_PATH = os.path.join(BASE_DIR, "backend", "metrics_report.txt")

# Training Config
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", 
    "SPY", "QQQ", "BTC-USD", "ETH-USD", "JPM", "BAC", "WMT"
]

# Features to use (same as quant model but maybe less is ok)
FEATURES = [
    'vol_10', 'sma_10', 'sma_50', 'ema_12', 'ema_26', 
    'macd', 'macd_signal', 'rsi_14', 'atr_14', 
    'stoch_k', 'stoch_d', 'obv', 
    'bb_width', 'roc_10', 'adx_14',
    'return_lag1', 'return_lag2', 'return_lag3',
    'vol_lag1', 'rsi_lag1', 'macd_lag1'
]

def train_regression_model():
    print(f"\n=== Training Daily Regression Model ===")
    
    period = "2y"
    interval = "1d"
    prediction_horizon = 1
    
    all_data = []
    
    for ticker in TICKERS:
        print(f"Fetching {interval} data for {ticker}...")
        try:
            df = utils.fetch_price_data(ticker, period=period, interval=interval)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue
            
        if df is None or df.empty:
            print(f"Skipping {ticker}: No data")
            continue
            
        # Add Features
        try:
            df = utils.add_technical_features(df)
        except Exception as e:
            print(f"Feature error {ticker}: {e}")
            continue
        
        # Create Target (Future Return)
        df['target'] = df['return'].shift(-prediction_horizon)
        
        df = df.dropna()
        
        # Clip outliers for regression stability
        # e.g. cap returns at +/- 20% for daily stocks (crypto might be higher but avg is lower)
        df['target'] = df['target'].clip(-0.2, 0.2)
        
        all_data.append(df)
        
    if not all_data:
        print("❌ No data collected. Aborting.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Total samples: {len(full_df)}")
    
    # Valid features
    available_features = [f for f in FEATURES if f in full_df.columns]
    X = full_df[available_features]
    y = full_df['target']
    
    # Check for NaNs
    X = X.fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter Tuning
    print("Tuning hyperparameters...")
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }
    
    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    search = RandomizedSearchCV(
        xgb, 
        param_distributions=param_grid, 
        n_iter=10, 
        scoring='neg_mean_squared_error',
        cv=3, 
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train_scaled, y_train)
    best_model = search.best_estimator_
    print(f"Best Params: {search.best_params_}")
    
    # Evaluate
    preds = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    print(f"Model RMSE: {rmse:.5f} (approx {rmse*100:.2f}%)")
    print(f"Model R2: {r2:.5f}")
    
    # Save
    joblib.dump(best_model, MODEL_DAILY_PATH)
    joblib.dump(scaler, SCALER_DAILY_PATH)
    joblib.dump(available_features, FEATURES_DAILY_PATH)
    
    print(f"Saved artifacts to {MODELS_DIR}")
    
    # Evaluate on test set for report
    with open(METRICS_PATH, "a") as f:
        f.write(f"\n--- Regression Model ---\n")
        f.write(f"RMSE: {rmse:.5f}\n")
        f.write(f"R2: {r2:.5f}\n")

if __name__ == "__main__":
    train_regression_model()

import os
import sys
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score

# Add backend to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import utils
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(os.getcwd(), 'backend'))
    import utils

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Artifact Paths
MODEL_DAILY_PATH = os.path.join(MODELS_DIR, "model_quant_daily.pkl")
SCALER_DAILY_PATH = os.path.join(MODELS_DIR, "scaler_quant_daily.pkl")
FEATURES_DAILY_PATH = os.path.join(MODELS_DIR, "features_quant_daily.pkl")

MODEL_INTRADAY_PATH = os.path.join(MODELS_DIR, "model_quant_intraday.pkl")
SCALER_INTRADAY_PATH = os.path.join(MODELS_DIR, "scaler_quant_intraday.pkl")
FEATURES_INTRADAY_PATH = os.path.join(MODELS_DIR, "features_quant_intraday.pkl")

# Training Config
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", 
    "SPY", "QQQ", "BTC-USD", "ETH-USD", "JPM", "BAC", "WMT"
]

# Features to use
FEATURES = [
    'vol_10', 'sma_10', 'sma_50', 'ema_12', 'ema_26', 
    'macd', 'macd_signal', 'rsi_14', 'atr_14', 
    'stoch_k', 'stoch_d', 'obv', 
    'bb_width', 'roc_10', 'adx_14', 'plus_di', 'minus_di',
    'return_lag1', 'return_lag2', 'return_lag3',
    'vol_lag1', 'rsi_lag1', 'macd_lag1',
    'Doji', 'Hammer', 'Bullish_Engulfing', 'Bearish_Engulfing'
]

def get_signal_label(pct_change, threshold=0.01):
    if pct_change >= threshold:
        return 2 # BUY
    elif pct_change <= -threshold:
        return 0 # SELL
    else:
        return 1 # HOLD/NEUTRAL

def train_model(model_type="daily"):
    print(f"\n=== Training {model_type.capitalize()} Quant Model ===")
    
    period = "2y" if model_type == "daily" else "60d"
    interval = "1d" if model_type == "daily" else "5m" # 1h might be better for stability, but 5m requested
    threshold = 0.01 if model_type == "daily" else 0.002 # 1% for daily, 0.2% for intraday
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
        
        # Create Target
        df['future_return'] = df['return'].shift(-prediction_horizon)
        df['target'] = df['future_return'].apply(lambda x: get_signal_label(x, threshold))
        
        df = df.dropna()
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
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Handle Class Imbalance with Sample Weights
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights_train = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    print("Class weights applied to balance training.")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter Tuning
    # Hyperparameter Tuning
    print("Tuning hyperparameters...")
    
    if model_type == "daily":
        # Deeper search for Daily
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5],
            'min_child_weight': [1, 3, 5]
        }
        n_iter = 15
        cv = 3
    else:
        # Faster search for Intraday demo
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        n_iter = 2
        cv = 2
    
    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # Use RandomizedSearchCV
    
    search = RandomizedSearchCV(
        xgb, 
        param_distributions=param_grid, 
        n_iter=n_iter, 
        scoring='f1_weighted',  # Optimize for F1 instead of Accuracy
        cv=cv, 
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit with weights
    search.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    
    best_model = search.best_estimator_
    print(f"Best Params: {search.best_params_}")
    
    # Feature Importance
    importance = best_model.feature_importances_
    feature_importance = pd.DataFrame({'feature': available_features, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Evaluate
    preds = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted')
    
    print(f"Model Accuracy: {acc*100:.2f}%")
    print(f"Model Precision: {precision*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['SELL', 'HOLD', 'BUY']))
    
    # Save
    if model_type == "daily":
        joblib.dump(best_model, MODEL_DAILY_PATH)
        joblib.dump(scaler, SCALER_DAILY_PATH)
        joblib.dump(available_features, FEATURES_DAILY_PATH)
    else:
        joblib.dump(best_model, MODEL_INTRADAY_PATH)
        joblib.dump(scaler, SCALER_INTRADAY_PATH)
        joblib.dump(available_features, FEATURES_INTRADAY_PATH)
        
    print(f"Saved artifacts to {MODELS_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["daily", "intraday", "all"], default="all", help="Model type to train")
    args = parser.parse_args()
    
    if args.type in ["daily", "all"]:
        train_model("daily")
    if args.type in ["intraday", "all"]:
        train_model("intraday")

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
MODEL_DAILY_PATH = os.path.join(MODELS_DIR, "model_daily.pkl")
SCALER_DAILY_PATH = os.path.join(MODELS_DIR, "scaler_model_daily.pkl")
FEATURES_DAILY_PATH = os.path.join(MODELS_DIR, "features_model_daily.pkl")

def evaluate_daily_model():
    print("=== Evaluating Daily Model ===")
    
    # Load Model
    if not os.path.exists(MODEL_DAILY_PATH):
        print(f"Error: Model not found at {MODEL_DAILY_PATH}")
        return

    try:
        model = joblib.load(MODEL_DAILY_PATH)
        scaler = joblib.load(SCALER_DAILY_PATH)
        features_list = joblib.load(FEATURES_DAILY_PATH)
        print("[OK] Model, Scaler, and Features loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load model artifacts: {e}")
        return
    
    tickers = ["AAPL", "NVDA", "SPY", "BTC-USD", "TSLA", "AMD"]
    all_y_true = []
    all_y_pred = []
    
    print(f"Testing on tickers: {', '.join(tickers)}")
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        # Fetch 1y data to ensure we have enough for lag features and a test set
        df = utils.fetch_price_data(ticker, period="1y")
        if df is None or df.empty:
            print(f"Skipping {ticker}: No data")
            continue
            
        df = utils.add_technical_features(df)
        
        # Create Target: Next Day Return
        # We want to predict tomorrow's return based on today's data.
        # So Target = df['return'].shift(-1)
        df['target'] = df['return'].shift(-1)
        
        # Drop NaNs (last row will have NaN target, first rows have NaN features)
        df = df.dropna()
        
        if len(df) < 60:
            print(f"Skipping {ticker}: Not enough data after processing ({len(df)} rows)")
            continue
            
        # Split into Test Set (Last 60 days)
        test_df = df.tail(60).copy()
        
        # Prepare Features
        # Add macro data (using current for simplicity - NOTE: This is a simplification)
        macro_data = utils.get_macro_data()
        for key, val in macro_data.items():
            test_df[key] = val
            
        # Add sentiment (constant for now)
        test_df['sentiment'] = 0.5 
        
        # Ensure columns exist and match training features
        for col in features_list:
            if col not in test_df.columns:
                test_df[col] = 0.0
                
        X_test = test_df[features_list]
        y_test = test_df['target']
        
        # Scale
        try:
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"Scaling error for {ticker}: {e}")
            continue
        
        # Predict
        try:
            preds = model.predict(X_test_scaled)
        except Exception as e:
            print(f"Prediction error for {ticker}: {e}")
            continue
        
        # Store
        all_y_true.extend(y_test.values)
        all_y_pred.extend(preds)
        
        # Per-ticker metrics
        mae = mean_absolute_error(y_test, preds)
        
        # Directional Accuracy
        # Sign of return: +1 for positive, -1 for negative (0 is rare but handled)
        # We check if sign matches
        correct_direction = np.sign(y_test.values) == np.sign(preds)
        dir_acc = np.mean(correct_direction)
        
        print(f"  -> MAE: {mae:.4f}")
        print(f"  -> Directional Accuracy: {dir_acc*100:.2f}%")
        
        # Show a few examples
        print("  -> Sample Predictions (Pred vs Actual):")
        for i in range(min(3, len(preds))):
            print(f"     {preds[i]*100:.2f}% vs {y_test.values[i]*100:.2f}%")

    # Overall Metrics
    if all_y_true:
        print("\n=== Overall Results ===")
        overall_mae = mean_absolute_error(all_y_true, all_y_pred)
        overall_mse = mean_squared_error(all_y_true, all_y_pred)
        overall_rmse = np.sqrt(overall_mse)
        
        correct_direction = np.sign(all_y_true) == np.sign(all_y_pred)
        overall_dir_acc = np.mean(correct_direction)
        
        print(f"Overall MAE: {overall_mae:.4f}")
        print(f"Overall RMSE: {overall_rmse:.4f}")
        print(f"Overall Directional Accuracy: {overall_dir_acc*100:.2f}%")
        
        print("\nInterpretation:")
        print(f"- MAE of {overall_mae:.4f} means on average the prediction is off by {overall_mae*100:.2f} percentage points.")
        print(f"- Directional Accuracy of {overall_dir_acc*100:.2f}% means the model correctly predicts Up/Down {overall_dir_acc*100:.1f}% of the time.")

        # Write to summary.txt
        with open("summary.txt", "w", encoding="utf-8") as f:
            f.write("=== Model Evaluation Summary ===\n")
            f.write(f"Overall MAE: {overall_mae:.4f}\n")
            f.write(f"Overall RMSE: {overall_rmse:.4f}\n")
            f.write(f"Overall Directional Accuracy: {overall_dir_acc*100:.2f}%\n")
            f.write("\nInterpretation:\n")
            f.write(f"- MAE of {overall_mae:.4f} means on average the prediction is off by {overall_mae*100:.2f} percentage points.\n")
            f.write(f"- Directional Accuracy of {overall_dir_acc*100:.2f}% means the model correctly predicts Up/Down {overall_dir_acc*100:.1f}% of the time.\n")
    else:
        print("No results generated.")

if __name__ == "__main__":
    evaluate_daily_model()

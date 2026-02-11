import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "all_tickers_with_macro_sentiment.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.pkl")

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # === Define features dynamically (exclude Date, Ticker, target) ===
    FEATURES = [col for col in df.columns if col not in ["Date", "Ticker", "target"]]
    TARGET = "target"

    X = df[FEATURES]
    y = df[TARGET]

    # Train/test split (no shuffle because time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scaling (optional for RandomForest, but useful if you swap models later)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logging.info(f"✅ Test RMSE: {rmse:.4f}")
    logging.info(f"✅ R² Score: {r2:.4f}")

    # Save model, scaler, and features
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FEATURES, FEATURES_PATH)

    logging.info(f"🎯 Model saved to {MODEL_PATH}")
    logging.info(f"🎯 Scaler saved to {SCALER_PATH}")
    logging.info(f"🎯 Features saved to {FEATURES_PATH}")

if __name__ == "__main__":
    main()

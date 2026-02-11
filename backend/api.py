import os
import math
import requests
import joblib
import pandas as pd
import numpy as np
import subprocess
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import create_model, BaseModel
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from passlib.context import CryptContext
import bcrypt

# --- Passlib + Bcrypt 4.1.0 Compatibility Patch ---
# passlib release (1.7.4) predates bcrypt 4.1.0 which removed __about__
# We explicitly patch it to avoid AttributeError.
try:
    import passlib.handlers.bcrypt
    passlib.handlers.bcrypt.bcrypt.version = bcrypt.__version__
except Exception:
    pass
# --------------------------------------------------

try:
    from . import utils
except ImportError:
    import utils
import google.generativeai as genai
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb

# === Load environment ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# MySQL config
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DB = os.getenv("MYSQL_DB", "stock_portfolio")

# Gemini Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ GEMINI_API_KEY not found. Generative AI features will be disabled.")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Connect to MySQL
try:
    db = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )
    cursor = db.cursor(dictionary=True)
except Exception as e:
    print(f"Warning: Database connection failed: {e}")
    db = None
    cursor = None

# FastAPI app
app = FastAPI(
    title="Portfolio Risk Analyzer API",
    description="Predicts next-day stock prices, trading signals, and handles user auth.",
    version="2.1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load model/scaler/features ===
# === Load model/scaler/features ===
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_DAILY_PATH = os.path.join(MODELS_DIR, "model_daily.pkl")
MODEL_INTRADAY_PATH = os.path.join(MODELS_DIR, "model_intraday.pkl")
SCALER_DAILY_PATH = os.path.join(MODELS_DIR, "scaler_model_daily.pkl")
SCALER_INTRADAY_PATH = os.path.join(MODELS_DIR, "scaler_model_intraday.pkl")
FEATURES_DAILY_PATH = os.path.join(MODELS_DIR, "features_model_daily.pkl")
FEATURES_INTRADAY_PATH = os.path.join(MODELS_DIR, "features_model_intraday.pkl")
MODEL_LSTM_PATH = os.path.join(MODELS_DIR, "lstm_daily.h5")
SCALER_LSTM_PATH = os.path.join(MODELS_DIR, "scaler_lstm_daily_lstm.pkl")
FEATURES_LSTM_PATH = os.path.join(MODELS_DIR, "features_lstm_daily_lstm.pkl")
MODEL_SCALPER_PATH = os.path.join(MODELS_DIR, "model_scalper.pkl")
FEATURES_SCALPER_PATH = os.path.join(MODELS_DIR, "features_model_scalper.pkl")

# Quant Models
MODEL_QUANT_DAILY_PATH = os.path.join(MODELS_DIR, "model_quant_daily.pkl")
SCALER_QUANT_DAILY_PATH = os.path.join(MODELS_DIR, "scaler_quant_daily.pkl")
FEATURES_QUANT_DAILY_PATH = os.path.join(MODELS_DIR, "features_quant_daily.pkl")

MODEL_QUANT_INTRADAY_PATH = os.path.join(MODELS_DIR, "model_quant_intraday.pkl")
SCALER_QUANT_INTRADAY_PATH = os.path.join(MODELS_DIR, "scaler_quant_intraday.pkl")
FEATURES_QUANT_INTRADAY_PATH = os.path.join(MODELS_DIR, "features_quant_intraday.pkl")

models = {}
scalers = {}
features_list = {}

def load_models():
    try:
        models['daily'] = joblib.load(MODEL_DAILY_PATH)
        scalers['daily'] = joblib.load(SCALER_DAILY_PATH)
        features_list['daily'] = joblib.load(FEATURES_DAILY_PATH)
        print("✅ Daily Model loaded.")
    except Exception as e:
        print(f"⚠️ Daily Model not found: {e}")

    try:
        models['intraday'] = joblib.load(MODEL_INTRADAY_PATH)
        scalers['intraday'] = joblib.load(SCALER_INTRADAY_PATH)
        features_list['intraday'] = joblib.load(FEATURES_INTRADAY_PATH)
        print("✅ Intraday Model loaded.")
    except Exception as e:
        print(f"⚠️ Intraday Model not found: {e}")

    try:
        if os.path.exists(MODEL_LSTM_PATH):
            models['lstm'] = load_model(MODEL_LSTM_PATH)
            scalers['lstm'] = joblib.load(SCALER_LSTM_PATH)
            features_list['lstm'] = joblib.load(FEATURES_LSTM_PATH)
            print("✅ LSTM Model loaded.")
    except Exception as e:
        print(f"⚠️ LSTM Model not found: {e}")

    try:
        if os.path.exists(MODEL_SCALPER_PATH):
            models['scalper'] = joblib.load(MODEL_SCALPER_PATH)
            features_list['scalper'] = joblib.load(FEATURES_SCALPER_PATH)
            print("✅ Scalper Model loaded.")
    except Exception as e:
        print(f"⚠️ Scalper Model not found: {e}")

    try:
        if os.path.exists(MODEL_QUANT_DAILY_PATH):
            models['quant_daily'] = joblib.load(MODEL_QUANT_DAILY_PATH)
            scalers['quant_daily'] = joblib.load(SCALER_QUANT_DAILY_PATH)
            features_list['quant_daily'] = joblib.load(FEATURES_QUANT_DAILY_PATH)
            print("✅ Quant Daily Model loaded.")
    except Exception as e:
        print(f"⚠️ Quant Daily Model not found: {e}")

    try:
        if os.path.exists(MODEL_QUANT_INTRADAY_PATH):
            models['quant_intraday'] = joblib.load(MODEL_QUANT_INTRADAY_PATH)
            scalers['quant_intraday'] = joblib.load(SCALER_QUANT_INTRADAY_PATH)
            features_list['quant_intraday'] = joblib.load(FEATURES_QUANT_INTRADAY_PATH)
            print("✅ Quant Intraday Model loaded.")
    except Exception as e:
        print(f"⚠️ Quant Intraday Model not found: {e}")

load_models()

# === Dynamically create Pydantic model for prediction ===
# We use a flexible dict for input to avoid validation errors if features change
class PredictionInput(BaseModel):
    features: dict

# === Prediction thresholds ===
BUY_THRESHOLD = 0.01
SELL_THRESHOLD = -0.01

# === Routes ===
@app.get("/")
async def root():
    return {"message": "Portfolio Risk Analyzer API is running 🚀"}

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "API alive and healthy ✅"}

# === User Registration/Login Models ===
class RegisterModel(BaseModel):
    name: str
    email: str
    password: str

class LoginModel(BaseModel):
    email: str
    password: str

# === User Registration ===
@app.post("/register")
def register(user: RegisterModel):
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")
    hashed_password = pwd_context.hash(user.password)
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
            (user.name, user.email, hashed_password)
        )
        db.commit()
        return {"success": True, "message": "User registered successfully"}
    except mysql.connector.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")

# === User Login ===
@app.post("/login")
def login(user: LoginModel):
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")
    cursor.execute("SELECT * FROM users WHERE email=%s", (user.email,))
    db_user = cursor.fetchone()
    if db_user and pwd_context.verify(user.password, db_user["password_hash"]):
        return {
            "success": True,
            "message": "Login successful",
            "user": {"name": db_user["name"], "email": db_user["email"]}
        }
    raise HTTPException(status_code=400, detail="Invalid email or password")

# === Prediction Endpoint (Single) ===
@app.post("/predict")
def predict(input_data: PredictionInput):
    if 'daily' not in models:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Ensure all required features are present
        features_dict = input_data.features
        # Fill missing features with 0.0 if necessary (or fail)
        input_df = pd.DataFrame([features_dict])
        
        # Check for missing columns
        required_feats = features_list['daily']
        missing_cols = [col for col in required_feats if col not in input_df.columns]
        if missing_cols:
            # Try to calculate them if possible? No, single point prediction assumes features provided.
            # Just fill with 0
            for col in missing_cols:
                input_df[col] = 0.0
        
        input_df = input_df[required_feats]
        
        # Validate input: Check if all features are zero (suspicious)
        if (input_df == 0).all().all():
             raise HTTPException(status_code=400, detail="Insufficient data for prediction (all features zero)")
             
        # Check for NaNs
        if input_df.isnull().values.any():
             # Fill with 0 as fallback
             input_df = input_df.fillna(0.0)

        input_scaled = scalers['daily'].transform(input_df)
        pred = models['daily'].predict(input_scaled)
        predicted_price = float(np.squeeze(pred))
        current_price = float(features_dict.get("Close", predicted_price))
        
        pct_change = (predicted_price - current_price) / current_price if current_price != 0 else 0
        pct_change_percent = pct_change * 100
        conf = 1 / (1 + math.exp(-pct_change * 50))

        if pct_change >= BUY_THRESHOLD:
            signal = "BUY"
        elif pct_change <= SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "success": True,
            "signal": signal,
            "predicted_price": predicted_price,
            "current_price": current_price,
            "expected_change_percent": pct_change_percent,
            "confidence": conf
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# === Telegram Report Endpoint ===
class TelegramReportRequest(BaseModel):
    stocks: list

@app.post("/send_telegram_report")
def send_telegram_report(request: TelegramReportRequest):
    if not request.stocks:
        raise HTTPException(status_code=400, detail="No stock data provided")
    
    message = "📊 *Portfolio Risk Report* 📊\n\n"
    for stock in request.stocks:
        symbol = stock.get('symbol', 'N/A')
        price = stock.get('price', 0)
        prediction = stock.get('prediction', 'N/A')
        signal = stock.get('signal', 'N/A')
        risk = stock.get('risk', 'N/A')
        
        emoji = "🟢" if signal == "Buy" else "🔴" if signal == "Sell" else "🟡"
        
        message += f"{emoji} *{symbol}*\n"
        message += f"   Price: ${price}\n"
        message += f"   Pred: {prediction}\n"
        message += f"   Signal: {signal} | Risk: {risk}\n\n"
        
    message += "🚀 *Generated by Portfolio Risk Analyzer*"
    
    send_telegram_alert(message)
    return {"success": True, "message": "Telegram report sent!"}

# === Live Predictions Endpoint ===
@app.get("/live_predictions")
def live_predictions():
    if 'daily' not in models:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "INTC", "NFLX", # Tech
        "JPM", "BAC", "V", "MA", "GS", # Finance
        "WMT", "PG", "KO", "DIS", "MCD", # Consumer
        "SPY", "QQQ", # Indices
        "BTC-USD", "ETH-USD", # Crypto
        
        # --- Indian Stocks (NSE) ---
        "^NSEI", # Nifty 50 Index
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", # Large Cap
        "TATASTEEL.NS", "TATAPOWER.NS", "TITAN.NS", # Tata Group
        "BEL.NS", "HAL.NS", # Defense / PSU
        "M&M.NS", # Auto
        "ADANIENT.NS", "SBIN.NS", "BHARTIARTL.NS" # Others
    ]
    results = []
    
    macro_data = utils.get_macro_data()
    
    for ticker in tickers:
        try:
            # Fetch data (enough for indicators)
            df = utils.fetch_price_data(ticker, period="6mo")
            if df is None or df.empty:
                continue
            
            # Add features
            df = utils.add_technical_features(df)
            if df.empty:
                continue
            
            latest = df.iloc[-1]
            current_price = float(latest['Close'])
            
            # Prepare features
            features_dict = latest.to_dict()
            features_dict.update(macro_data)
            features_dict['sentiment'] = utils.get_sentiment(ticker)
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([features_dict])
            
            # Ensure columns match model features
            required_feats = features_list['daily']
            for col in required_feats:
                if col not in input_df.columns:
                    input_df[col] = 0.0 
            
            input_df = input_df[required_feats]
            
            # Predict
            input_scaled = scalers['daily'].transform(input_df)
            pct_change = float(models['daily'].predict(input_scaled)[0])
            pred_price = float(latest['Close']) * (1 + pct_change)
            
            # Determine Signal (Align with Quant Model if available)
            signal = "Hold"
            recommendation = "Hold"
            confidence_val = 0.0
            
            if 'quant_daily' in models:
                # Use Quant Model for Signal alignment
                q_input = pd.DataFrame([features_dict])
                q_feats = features_list['quant_daily']
                for c in q_feats:
                    if c not in q_input.columns: q_input[c] = 0.0
                q_scaled = scalers['quant_daily'].transform(q_input[q_feats])
                probs = models['quant_daily'].predict_proba(q_scaled)[0] # [Sell, Hold, Buy]
                
                prob_sell, prob_hold, prob_buy = probs[0], probs[1], probs[2]
                
                # Simplified Decision Engine Logic for Dashboard
                if prob_buy > 0.45 and prob_buy > prob_sell:
                    signal = "Buy"
                    recommendation = "Buy" if prob_buy < 0.6 else "Strong Buy"
                    confidence_val = prob_buy
                elif prob_sell > 0.45 and prob_sell > prob_buy:
                    signal = "Sell"
                    recommendation = "Sell" if prob_sell < 0.6 else "Strong Sell"
                    confidence_val = prob_sell
                else:
                    signal = "Hold"
                    confidence_val = prob_hold
            else:
                # Fallback to Regression Thresholds
                if pct_change >= 0.01:
                    signal = "Buy"
                    recommendation = "Strong Buy"
                    confidence_val = abs(pct_change) * 50
                elif pct_change <= -0.01:
                    signal = "Sell"
                    recommendation = "Strong Sell"
                    confidence_val = abs(pct_change) * 50
                
            # Risk based on Volatility
            vol = latest.get('vol_10', 0)
            if vol > 0.02:
                risk = "High"
            elif vol > 0.01:
                risk = "Medium"
            else:
                risk = "Low"
            
            # Chart data: Last 30 days history
            history_df = df.tail(30).copy()
            history_data = []
            for _, row in history_df.iterrows():
                # Format date as string (e.g., "Mon", "Jan 01" or just ISO)
                # row['Date'] might be Timestamp
                date_str = row['Date'].strftime('%b %d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
                history_data.append({
                    "name": date_str,
                    "value": round(float(row['Close']), 2)
                })

            stock_result = {
                "symbol": ticker,
                "name": ticker, 
                "price": round(current_price, 2),
                "change": f"{pct_change*100:.2f}%",
                "prediction": f"${pred_price:.2f}",
                "signal": signal,
                "risk": risk,
                "recommendation": recommendation,
                "confidence": f"{confidence_val*100:.1f}%",
                "history": history_data
            }
            results.append(stock_result)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # === Aggregate Portfolio History for Chart ===
    # We will sum the 'Close' prices of all stocks for each date to simulate a portfolio
    portfolio_history = {}
    
    for stock in results:
        for point in stock['history']:
            date = point['name']
            value = point['value']
            if date in portfolio_history:
                portfolio_history[date] += value
            else:
                portfolio_history[date] = value
                
    # Convert to list of dicts and sort by date (assuming date strings are sortable or consistent)
    # Since we formatted dates as "Jan 01", sorting might be tricky if spanning years. 
    # For now, we'll rely on the order of the first stock's history which is chronological.
    
    chart_data = []
    if results:
        # Use the dates from the first stock to ensure order
        reference_history = results[0]['history']
        for point in reference_history:
            date = point['name']
            if date in portfolio_history:
                chart_data.append({
                    "name": date,
                    "value": round(portfolio_history[date], 2)
                })

    return {"stocks": results, "chartData": chart_data}

# === Single Stock Real-time Analysis ===
import random
from datetime import datetime, timedelta

# Simple in-memory cache: {ticker: {'timestamp': time, 'data': dict}}
STOCK_CACHE = {}
CACHE_DURATION = 60  # seconds

def get_next_trading_day():
    today = datetime.now()
    # If Friday (4), Saturday (5), or Sunday (6), next is Monday
    if today.weekday() >= 4:
        days_ahead = 7 - today.weekday() # Fri(4)->3, Sat(5)->2, Sun(6)->1 ... wait. 
        # Fri(4) + 3 = Mon(0). Correct.
        # Sat(5) + 2 = Mon(0). Correct.
        # Sun(6) + 1 = Mon(0). Correct.
        next_day = today + timedelta(days=days_ahead)
    else:
        next_day = today + timedelta(days=1)
    return next_day.strftime('%Y-%m-%d')

@app.get("/analyze_stock/{ticker}")
def analyze_stock(ticker: str):
    try:
        from .decision_engine import DecisionEngine
        from .risk_engine import RiskEngine
        from .orchestrator import Orchestrator
    except ImportError:
        from decision_engine import DecisionEngine
        from risk_engine import RiskEngine
        from orchestrator import Orchestrator

    decision_engine = DecisionEngine()

    try:
        # --- 1. Gather Data ---
        # Fetch Daily Data (Swing)
        df_daily = utils.fetch_price_data(ticker, period="1y") 
        if df_daily is None or df_daily.empty:
             raise HTTPException(status_code=404, detail=f"Data not found for {ticker}")
        
        df_daily = utils.add_technical_features(df_daily)
        latest_daily = df_daily.iloc[-1]
        current_price = float(latest_daily['Close'])
        
        # Fetch Intraday Data
        df_intra = utils.fetch_price_data(ticker, period="5d", interval="5m")
        if df_intra is not None and not df_intra.empty:
            df_intra = utils.add_technical_features(df_intra)
            latest_intra = df_intra.iloc[-1]
        else:
            latest_intra = None

        # --- 2. Get Model Probabilities ---
        
        # Daily Probs
        daily_probs = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        if 'quant_daily' in models:
            features_dict = latest_daily.to_dict()
            features_dict.update(utils.get_macro_data()) 
            features_dict['sentiment'] = utils.get_sentiment(ticker)
            
            input_df = pd.DataFrame([features_dict])
            required_feats = features_list['quant_daily']
            for col in required_feats:
                if col not in input_df.columns: input_df[col] = 0.0
            
            input_scaled = scalers['quant_daily'].transform(input_df[required_feats])
            probs = models['quant_daily'].predict_proba(input_scaled)[0] # [Sell, Hold, Buy]
            daily_probs = {'sell': probs[0], 'hold': probs[1], 'buy': probs[2]}

        # Intraday Probs
        intraday_probs = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        if latest_intra is not None and 'quant_intraday' in models:
            features_intra = latest_intra.to_dict()
            input_intra = pd.DataFrame([features_intra])
            required_intra = features_list['quant_intraday']
            for col in required_intra:
                if col not in input_intra.columns: input_intra[col] = 0.0
                
            input_scaled_intra = scalers['quant_intraday'].transform(input_intra[required_intra])
            probs_i = models['quant_intraday'].predict_proba(input_scaled_intra)[0]
            intraday_probs = {'sell': probs_i[0], 'hold': probs_i[1], 'buy': probs_i[2]}

        # LSTM Prediction
        lstm_pred = 0.0
        if 'lstm' in models and len(df_daily) >= 60:
            try:
                lstm_feats = features_list['lstm']
                lstm_df = df_daily.tail(60).copy()
                for col in lstm_feats:
                    if col not in lstm_df.columns: lstm_df[col] = 0.0
                lstm_scaled = scalers['lstm'].transform(lstm_df[lstm_feats])
                lstm_input = np.array([lstm_scaled])
                lstm_pred = float(models['lstm'].predict(lstm_input, verbose=0)[0][0])
            except: pass

        # Scalper Signal (using existing logic or simplified)
        scalper_signal = "Neutral"
        # We can re-use monitor_trade logic or just pass a simple check
        # For efficiency, let's just say "Neutral" unless we call the dedicated scalper endpoint logic
        # Or quickly check scalper model here if loaded
        if 'scalper' in models and latest_intra is not None:
             # reuse input_intra
             # ... (scalper prediction logic if needed, or abstract it)
             pass

        # --- 3. Run Production Pipeline (Orchestrator) ---
        
        # Prepare Market Data for Orchestrator
        # Technicals
        technicals = {
            "rsi_14": float(latest_daily.get('rsi_14', 50)),
            "adx_14": float(latest_daily.get('adx_14', 20)),
            "bb_width": float(latest_daily.get('bb_width', 0.1)),
            "macd": float(latest_daily.get('macd', 0)),
            "atr_14": float(latest_daily.get('atr_14', current_price*0.02))
        }
        
        # Volatility
        vol = float(latest_daily.get('vol_10', 0.02))
        vol_type = "High" if vol > 0.02 else "Medium" if vol > 0.01 else "Low"
        
        # Sentiment & Macro
        sentiment = utils.get_sentiment(ticker)
        macro_data = utils.get_macro_data()
        macro = macro_data.get('trend', 'Neutral') 

        # Regression Prediction (Price Change)
        reg_pred = 0.0
        if 'daily' in models:
             reg_dict = latest_daily.to_dict()
             reg_dict.update(macro_data)
             reg_dict['sentiment'] = sentiment
             reg_input = pd.DataFrame([reg_dict])
             reg_feats = features_list['daily']
             for c in reg_feats: 
                 if c not in reg_input.columns: reg_input[c] = 0.0
             reg_scaled = scalers['daily'].transform(reg_input[reg_feats])
             reg_pred = float(models['daily'].predict(reg_scaled)[0])

        # Load Confidence History
        CONFIDENCE_HISTORY_FILE = os.path.join(BASE_DIR, "data", "confidence_history.json")
        import json
        history_scores = []
        if os.path.exists(CONFIDENCE_HISTORY_FILE):
             try:
                 with open(CONFIDENCE_HISTORY_FILE, "r") as f:
                     h = json.load(f)
                     history_scores = h.get(ticker, [])
             except: pass

        market_data = {
            "current_price": current_price,
            "daily_probs": daily_probs, 
            "intraday_probs": intraday_probs, 
            "scalper_signal": scalper_signal,
            "lstm_pred": lstm_pred,
            "regression_pred": reg_pred,
            "volatility_type": vol_type,
            "technicals": technicals,
            "sentiment_score": sentiment,
            "macro_trend": macro,
            "asset_type": "Crypto" if "-USD" in ticker else "Stock",
            "confidence_history": history_scores
        }
        
        # Execute Pipeline
        orchestrator = Orchestrator()
        pipeline_result = orchestrator.run_pipeline(ticker, market_data)
        
        # Save Confidence Score (Post-Pipeline)
        if pipeline_result['execution_permission'] or pipeline_result['final_action'] == 'HOLD':
             dec = pipeline_result.get('decision_output', {})
             if 'confidence_score' in dec:
                 new_score = dec['confidence_score']
                 history_scores.append(new_score)
                 if len(history_scores) > 100: history_scores = history_scores[-100:]
                 try:
                     full_hist = {}
                     if os.path.exists(CONFIDENCE_HISTORY_FILE):
                         with open(CONFIDENCE_HISTORY_FILE, "r") as f: full_hist = json.load(f)
                     full_hist[ticker] = history_scores
                     with open(CONFIDENCE_HISTORY_FILE, "w") as f: json.dump(full_hist, f)
                 except Exception as ex: print(f"History Save Fail: {ex}")

        # Format Response
        history_data = [] 
        for _, row in df_daily.tail(60).iterrows():
            date_iso = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
            history_data.append({"time": date_iso, "value": round(float(row['Close']), 2)})

        mapped_result = {
            "symbol": ticker,
            "price": round(current_price, 2),
            "history": history_data,
            "signal": pipeline_result['final_action'],
            "risk": pipeline_result.get('decision_output', {}).get('risk_level', 'Medium'),
            "confidence": pipeline_result.get('confidence', '0%'),
            "key_indicators": {
                "RSI": round(technicals['rsi_14'], 1),
                "MACD": round(technicals['macd'], 3),
                "Volatility": f"{vol*100:.1f}%",
                "SMA_50": round(latest_daily.get('sma_50', 0), 2),
                "Sentiment": f"{sentiment:.2f}"
            },
            "trade_params": pipeline_result.get('risk_output', {}),
            "validator_status": pipeline_result.get('validator_output', {}), 
            "orchestrator": {
                "status": pipeline_result['pipeline_status'],
                "block_reason": pipeline_result['block_reason']
            }
        }
        
        STOCK_CACHE[ticker] = {'timestamp': datetime.now(), 'data': mapped_result}
        return mapped_result

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# === Live Monitor Control ===
monitor_process = None

@app.post("/run-live-monitor")
def run_live_monitor():
    global monitor_process
    try:
        if monitor_process and monitor_process.poll() is None:
            return {"message": "⚠️ Live monitor is already running!"}
        script_path = os.path.join(BASE_DIR, "backend", "scripts", "live_monitor.py")
        monitor_process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return {"message": "✅ Live monitor started successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start live monitor: {str(e)}")

@app.get("/stock_price/{ticker}")
def get_stock_price(ticker: str):
    ticker = ticker.replace('$', '').upper() # Sanitize input
    price = utils.get_live_price(ticker)
    if price is None:
        raise HTTPException(status_code=404, detail="Price not found")
    return {"symbol": ticker, "price": round(price, 2)}

@app.get("/monitor-status")
def monitor_status():
    global monitor_process
    if monitor_process is None:
        return {"running": False, "message": "❌ Live monitor not started"}
    if monitor_process.poll() is None:
        return {"running": True, "message": "🟢 Live monitor is running"}
    return {"running": False, "message": "⚠️ Live monitor stopped"}

# === Real-Time Scalping Monitor ===
@app.get("/monitor_trade/{ticker}")
def monitor_trade(ticker: str):
    """
    Real-time trade monitoring endpoint.
    Returns ACTION: HOLD or SELL based on Scalper Model.
    """
    if 'scalper' not in models:
        raise HTTPException(status_code=500, detail="Scalper model not loaded")
        
    try:
        # Fetch latest 5m data (need enough for indicators)
        # yfinance 5m data is up to date usually
        df = utils.fetch_price_data(ticker, period="5d", interval="5m")
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found")
            
        # Add features
        df = utils.add_technical_features(df)
        latest = df.iloc[-1]
        
        # Prepare input
        features_dict = latest.to_dict()
        input_df = pd.DataFrame([features_dict])
        
        required_feats = features_list['scalper']
        for col in required_feats:
            if col not in input_df.columns: input_df[col] = 0.0
            
        input_df = input_df[required_feats]
        
        # Predict Direction (1=Up, 0=Down)
        probs = models['scalper'].predict_proba(input_df)[0] # [prob_down, prob_up]
        prob_down = probs[0]
        prob_up = probs[1]
        
        # Fetch LIVE price (Real-time, not 5m candle close)
        live_price = utils.get_live_price(ticker)
        if live_price is None:
            live_price = float(latest['Close']) # Fallback
        
        rsi = float(latest.get('rsi_14', 50))
        
        # Predictive Logic with Early Warning & Recommendations
        recommendation = ""
        
        if prob_down > 0.65:
            action = "STRONG SELL"
            reason = "High Probability of Drop"
            recommendation = "EXIT NOW - Protect Profits"
            color = "red"
        elif prob_down > 0.50:
            action = "SELL"
            reason = "Momentum Shifted Down"
            recommendation = "Sell Position Immediately"
            color = "red"
        elif prob_down > 0.35: # Early Warning Threshold
            action = "WARNING"
            reason = "Weakening Momentum (Risk)"
            recommendation = "PREPARE TO SELL - Watch Closely"
            color = "yellow"
        elif prob_up > 0.70 and rsi < 70:
            action = "STRONG BUY"
            reason = "Strong Upward Momentum"
            recommendation = "ENTER TRADE - Good Entry"
            color = "green"
        else:
            action = "HOLD / BUY"
            reason = "Trend is Positive"
            recommendation = "Hold Position / Add More"
            color = "green"
            
        confidence = max(prob_down, prob_up)
            
        return {
            "ticker": ticker,
            "price": live_price,
            "action": action,
            "reason": reason,
            "recommendation": recommendation,
            "confidence": f"{confidence*100:.1f}%",
            "color": color,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        return {
            "ticker": ticker,
            "price": live_price,
            "action": action,
            "reason": reason,
            "recommendation": recommendation,
            "confidence": f"{confidence*100:.1f}%",
            "color": color,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
    except Exception as e:
        print(f"Monitor Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Backtest Endpoint ===
@app.get("/backtest/{ticker}")
def run_backtest_endpoint(ticker: str, initial_capital: float = 10000.0):
    try:
        from .backtest_engine import BacktestEngine
    except ImportError:
        from backtest_engine import BacktestEngine

    # Fetch Data
    df = utils.fetch_price_data(ticker, period="1y")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data not found")
        
    # Add Features
    df = utils.add_technical_features(df)
    
    # Run Engine
    engine = BacktestEngine(
        df, 
        initial_capital=initial_capital,
        models=models,
        scalers=scalers,
        features_list=features_list
    )
    report = engine.run()
    
    return {
        "ticker": ticker,
        "metrics": report['trading_metrics'],
        "risk_metrics": report['risk_metrics'],
        "equity_curve": [{'time': i, 'value': v} for i, v in enumerate(engine.equity_curve)],
        "trades": engine.trades # Detailed trade logs
    }

# === Config Endpoint ===
class ConfigRequest(BaseModel):
    capital: float

@app.get("/config")
def get_config():
    # Read from .env or env vars
    cap = float(os.getenv("TRADING_CAPITAL", "10000.0"))
    return {"capital": cap}

@app.post("/config")
def update_config(req: ConfigRequest):
    try:
        new_cap = req.capital
        # Update current process env
        os.environ["TRADING_CAPITAL"] = str(new_cap)
        
        # Update .env file
        env_path = os.path.join(BASE_DIR, ".env")
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
        
        with open(env_path, "w") as f:
            found = False
            for line in lines:
                if line.startswith("TRADING_CAPITAL="):
                    f.write(f"TRADING_CAPITAL={new_cap}\n")
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(f"\nTRADING_CAPITAL={new_cap}\n")
                
        return {"success": True, "message": f"Capital updated to ${new_cap}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === AI Chatbot ===
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# Simple in-memory session store
# {session_id: {'last_ticker': str, 'last_intent': str}}
CHAT_SESSIONS = {}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    query = request.message.lower()
    session_id = request.session_id
    
    # Get context
    context = CHAT_SESSIONS.get(session_id, {'last_ticker': None, 'last_intent': None})
    
    # 1. Identify Ticker
    found_ticker = None
    # Check for exact matches or common names
    # We'll check our known tickers list
    known_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "INTC", "NFLX",
        "JPM", "BAC", "V", "MA", "GS",
        "WMT", "PG", "KO", "DIS", "MCD",
        "SPY", "QQQ",
        "BTC-USD", "ETH-USD", "BTC", "ETH", "BITCOIN", "ETHEREUM"
    ]
    
    # Map common names to symbols
    name_map = {
        "bitcoin": "BTC-USD",
        "btc": "BTC-USD",
        "ethereum": "ETH-USD",
        "eth": "ETH-USD",
        "google": "GOOGL",
        "amazon": "AMZN",
        "apple": "AAPL",
        "tesla": "TSLA",
        "microsoft": "MSFT",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "netflix": "NFLX"
    }
    
    words = query.split()
    for word in words:
        clean_word = word.strip("?!.,").upper()
        if clean_word in known_tickers:
            found_ticker = clean_word
            break
        if word.strip("?!.,").lower() in name_map:
            found_ticker = name_map[word.strip("?!.,").lower()]
            break
            
            found_ticker = name_map[word.strip("?!.,").lower()]
            break
            
    # Context Inference: Ticker
    if not found_ticker:
        if context['last_ticker']:
            found_ticker = context['last_ticker']
        else:
            return {
                "response": "I'm not sure which stock you're asking about. Try mentioning a ticker like AAPL, BTC, or MSFT.",
                "data": None
            }
        
    # Handle BTC/ETH shorthand
    if found_ticker == "BTC": found_ticker = "BTC-USD"
    if found_ticker == "ETH": found_ticker = "ETH-USD"

    # 2. Fetch Data
    try:
        data = analyze_stock(found_ticker)
    except Exception:
        return {"response": f"I couldn't fetch data for {found_ticker}. It might be invalid or down.", "data": None}
        
    price = data['price']
    signal_swing = data['signals']['swing']['signal']
    signal_intra = data['signals']['intraday']['signal']
    risk = data['risk']
    
    # 3. Determine Intent & Generate Response
    response = ""
    intent = "summary" # Default
    
    # Detect Intent
    if "price" in query or "cost" in query or "worth" in query:
        intent = "price"
    elif "buy" in query or "sell" in query or "signal" in query or "trade" in query or "should i" in query:
        intent = "signal"
    elif "risk" in query or "safe" in query or "volatility" in query:
        intent = "risk"
    elif "prediction" in query or "forecast" in query or "future" in query:
        intent = "prediction"
    elif "why" in query or "reason" in query or "explain" in query or "analysis" in query:
        intent = "reasoning"
    elif "teach" in query or "what is" in query or "how to" in query or "learn" in query:
        intent = "teaching"
    elif "open" in query or "close" in query or "status" in query or "time" in query:
        intent = "status"
    
    # Context Inference: Intent (if query is vague like "and for BTC?")
    # If we found a NEW ticker but no specific intent words, reuse the old intent
    if intent == "summary" and context['last_intent'] and found_ticker != context['last_ticker']:
        intent = context['last_intent']
        
    # Execute Intent
    if intent == "price":
        response = f"{found_ticker} is currently trading at ${price:,.2f}."
        
    elif intent == "signal":
        response = f"For {found_ticker}, the Swing (Daily) signal is **{signal_swing}** and the Intraday (5m) signal is **{signal_intra}**."
        if signal_swing == "Buy" and signal_intra == "Buy":
            response += " It looks like a strong buying opportunity!"
        elif signal_swing == "Sell" and signal_intra == "Sell":
            response += " Indicators suggest a selling trend."
            
    elif intent == "risk":
        response = f"{found_ticker} is currently rated as **{risk} Risk**. "
        if risk == "High":
            response += "Be careful, it's quite volatile right now."
        else:
            response += "It's showing relatively stable movement."
            
    elif intent == "prediction":
        pred_swing = data['signals']['swing']['prediction']
        pred_intra = data['signals']['intraday']['prediction']
        response = f"AI predicts {found_ticker} might move towards ${pred_swing:,.2f} (Daily) and ${pred_intra:,.2f} (Intraday)."
        
    elif intent == "status":
        status = data['market_status']
        response = f"The market for {found_ticker} is currently **{status}**."
        if status == "Open":
            response += " You can trade now! 🟢"
        else:
            response += " Trading is paused. 🔴"
            
    elif intent == "reasoning" or intent == "teaching":
        # Try Gemini first for reasoning/teaching
        gemini_reply = get_gemini_response(query, data)
        if gemini_reply:
            response = gemini_reply
        else:
            # Fallback to hardcoded logic if Gemini fails
            if intent == "reasoning":
                # Generate explanation based on indicators
                rsi = data['key_indicators']['RSI']
                macd = data['key_indicators']['MACD']
                vol_str = data['key_indicators']['Volatility']
                
                response = f"Here is the reasoning for {found_ticker}:\n"
                
                # RSI Logic
                if rsi > 70:
                    response += f"- **RSI is High ({rsi})**: It might be overbought (potential pullback).\n"
                elif rsi < 30:
                    response += f"- **RSI is Low ({rsi})**: It might be oversold (potential bounce).\n"
                else:
                    response += f"- **RSI is Neutral ({rsi})**: No extreme momentum.\n"
                    
                # MACD Logic
                if macd > 0:
                    response += f"- **MACD is Positive ({macd})**: Bullish momentum.\n"
                else:
                    response += f"- **MACD is Negative ({macd})**: Bearish momentum.\n"
                    
                # Signal Logic
                if signal_swing == "Buy":
                    response += f"- **AI Prediction**: Expects a price increase > 0.3%.\n"
                elif signal_swing == "Sell":
                    response += f"- **AI Prediction**: Expects a price drop > 0.3%.\n"
                else:
                    response += f"- **AI Prediction**: Expects flat movement (Hold).\n"
                    
                response += f"- **Volatility**: {vol_str} ({risk} Risk)."
            else:
                response = "I can't access my advanced teaching module right now (API Key missing). But I can tell you the signals!"
            
    else:
        # Default summary
        response = f"{found_ticker} is at ${price:,.2f}. Signals: Swing={signal_swing}, Intraday={signal_intra}. Risk={risk}."

    # Update Session
    CHAT_SESSIONS[session_id] = {'last_ticker': found_ticker, 'last_intent': intent}

    return {
        "response": response,
        "data": data # Frontend can use this to show a mini-card if needed
    }

# === Telegram Alert ===
def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram token or chat ID not set")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print("❌ Failed to send Telegram alert:", response.text)
    except Exception as e:
        print("❌ Error sending Telegram alert:", e)

# === Gemini Integration ===
def get_gemini_response(query: str, context_data: dict = None):
    """
    Generates a response using Google Gemini.
    """
    if not GEMINI_API_KEY:
        return None
        
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        You are an expert financial analyst and stock market assistant.
        User Query: "{query}"
        
        """
        
        if context_data:
            prompt += f"""
            Here is the real-time data for the stock mentioned:
            Symbol: {context_data.get('symbol')}
            Price: ${context_data.get('price')}
            Market Status: {context_data.get('market_status')}
            Risk Level: {context_data.get('risk')}
            
            Signals:
            - Swing (Daily): {context_data['signals']['swing']['signal']} (Action: {context_data['signals']['swing']['action']})
            - Intraday (5m): {context_data['signals']['intraday']['signal']}
            
            Technical Indicators:
            - RSI: {context_data['key_indicators']['RSI']}
            - MACD: {context_data['key_indicators']['MACD']}
            - Volatility: {context_data['key_indicators']['Volatility']}
            
            Use this data to answer the user. Explain the 'Why'. 
            If the user asks to 'teach', explain the concepts simply.
            Be professional but conversational. Keep it concise (under 100 words unless asked for detail).
            """
        else:
            prompt += "Answer general stock market questions. If you don't know, say so."
            
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

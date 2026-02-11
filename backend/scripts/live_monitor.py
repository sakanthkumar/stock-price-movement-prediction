import os
import time
import sys
import pandas as pd
import yfinance as yf
import requests
import joblib
import logging
from dotenv import load_dotenv

# ==========================================
# Setup & Imports
# ==========================================
# Add backend to path to import modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR) # e:/.../backend
PROJECT_DIR = os.path.dirname(BACKEND_DIR) # e:/.../

sys.path.append(BACKEND_DIR)

try:
    from utils import add_technical_features
    from risk_engine import RiskEngine
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(PROJECT_DIR, 'backend'))
    from utils import add_technical_features
    from risk_engine import RiskEngine

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================
# Configuration
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# Default to 10000 if not set
TRADING_CAPITAL = float(os.getenv("TRADING_CAPITAL", "10000.0")) 
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

# Initialize Risk Engine
risk_engine = RiskEngine(capital=TRADING_CAPITAL)

# ==========================================
# Telegram Messaging
# ==========================================
def send_telegram_message(message: str):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            r = requests.post(url, data=payload, timeout=10)
            if r.status_code != 200:
                logging.error(f"Telegram API error: {r.text}")
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {e}")
    else:
        logging.warning(f"⚠️ Telegram credentials missing! Msg: {message[:50]}...")

# ==========================================
# Data & Logic
# ==========================================
def fetch_and_analyze(ticker: str):
    try:
        # 1. Fetch Data (Need enough for indicators: 60+ days)
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=False)
        
        if df is None or df.empty:
            logging.warning(f"No data for {ticker}")
            return

        # Fix MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)

        # 2. Add Technical Indicators
        # We use the shared utils function to ensure consistency with Backtest
        df = add_technical_features(df)
        
        latest = df.iloc[-1]
        
        # 3. Simple Signal Logic (Trend Follow)
        # Replacing the old regression model with a robust technical check
        # Condition: Price > SMA_50 AND RSI < 70 (Uptrend but not overbought)
        # This aligns with the 'DecisionEngine' logic used in backtest
        
        close = latest['Close']
        sma_50 = latest.get('sma_50', close)
        rsi = latest.get('rsi_14', 50)
        vol = latest.get('vol_10', 0.02)
        atr = latest.get('atr_14', close * 0.02)
        
        signal = "HOLD"
        confidence = 50
        
        # Simple Logic: Bull Trend
        if close > sma_50 and rsi > 45 and rsi < 75:
            signal = "BUY"
            confidence = 75 if (vol < 0.03) else 65
            
        # 4. Apply Risk Engine
        if signal == "BUY":
            vol_type = "High" if vol > 0.02 else "Low"
            conf_level = "High" if confidence > 70 else "Medium"
            
            risk_params = risk_engine.calculate_trade_params(
                entry_price=close,
                atr=atr,
                confidence_level=conf_level,
                volatility_type=vol_type,
                horizon="Swing",
                capital=TRADING_CAPITAL
            )
            
            if risk_params['allowed']:
                # Send Alert
                size = risk_params['position_size']
                sl = risk_params['stop_loss']
                tp = risk_params['take_profit']
                
                msg = (
                    f"🚀 **BUY SIGNAL: {ticker}**\n"
                    f"Price: ${close:.2f}\n"
                    f"--------------------\n"
                    f"📦 **Size:** {size:.2f} shares\n"
                    f"🛑 **Stop Loss:** ${sl:.2f}\n"
                    f"🎯 **Take Profit:** ${tp:.2f}\n"
                    f"--------------------\n"
                    f"Confidence: {confidence}%\n"
                    f"Risk/Reward: 1:{((tp-close)/(close-sl)):.1f}"
                )
                logging.info(f"Triggering Alert for {ticker}")
                send_telegram_message(msg)
            else:
                logging.info(f"{ticker}: BUY Signal rejected by Risk Engine: {risk_params.get('reason')}")
        else:
             logging.info(f"{ticker}: Neutral/Hold (RSI: {rsi:.1f}, $ > SMA: {close > sma_50})")

    except Exception as e:
        logging.error(f"Error analyzing {ticker}: {e}")

# ==========================================
# Main Loop
# ==========================================
def main():
    logging.info("✅ Advanced Live Monitor Started (Risk Engine Integrated)")
    while True:
        # Market Hours Check (Optional, skipping for now to allow 24/7 crypto/testing)
        for ticker in TICKERS:
            fetch_and_analyze(ticker)
            
        logging.info("💤 Cycle complete. Waiting 15 minutes...")
        time.sleep(900) # 15 min interval

if __name__ == "__main__":
    main()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import utils
from backend.backtest_engine import BacktestEngine
import pandas as pd

def main():
    ticker = "NVDA"
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        
    print(f"Fetching data for {ticker}...")
    df = utils.fetch_price_data(ticker, period="1y")
    if df is None:
        print("Error fetching data")
        return
        
    print("Adding indicators...")
    df = utils.add_technical_features(df)
    
    engine = BacktestEngine(df, initial_capital=10000)
    report = engine.run()
    
    print("\n" + "="*40)
    print(f"BACKTEST REPORT: {ticker}")
    print("="*40)
    
    t_metrics = report['trading_metrics']
    r_metrics = report['risk_metrics']
    
    print(f"Total Trades: {t_metrics['total_trades']}")
    print(f"Win Rate:     {t_metrics['win_rate']}")
    print(f"Net Profit:   ${t_metrics['net_profit']}")
    print(f"Profit Factor:{t_metrics['profit_factor']}")
    print("-" * 20)
    print(f"Max Drawdown: {r_metrics['max_drawdown']}")
    print(f"Sharpe Ratio: {r_metrics['sharpe_ratio']}")
    print("="*40)

if __name__ == "__main__":
    main()

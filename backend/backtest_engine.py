import pandas as pd
import numpy as np
try:
    from .utils import add_technical_features
    from .decision_engine import DecisionEngine
    from .risk_engine import RiskEngine
    from .metrics_engine import MetricsEngine
except ImportError:
    from utils import add_technical_features
    from decision_engine import DecisionEngine
    from risk_engine import RiskEngine
    from metrics_engine import MetricsEngine
import logging

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, initial_capital=10000.0, models=None, scalers=None, features_list=None):
        self.data = data.copy() # Expect OHLCV
        self.capital = initial_capital
        self.decision_engine = DecisionEngine()
        self.risk_engine = RiskEngine(capital=initial_capital)
        self.metrics_engine = MetricsEngine()
        
        self.models = models or {}
        self.scalers = scalers or {}
        self.features_list = features_list or {}
        
        self.trades = [] # List of closed trades
        self.equity_curve = [initial_capital]
        self.position = None # None or dict { 'entry': float, 'size': float, 'sl': float, 'tp': float }
        
    def _format_date(self, idx):
        # Convert index to string 'YYYY-MM-DD' if it's timestamp
        try:
            return str(idx).split(' ')[0]
        except:
            return str(idx)

    def run(self):
        """
        Run the simulation row by row using Vectorized AI Inference.
        """
        print(f"Starting Backtest on {len(self.data)} bars...")
        
        # 0. Pre-compute AI Signals (Vectorized)
        self._precompute_ai_signals()
        
        # Track equity curve with dates
        self.equity_series = [] # [{'time': '2023...', 'value': 10000}]
        
        start_idx = 50 # Warmup for indicators
        
        # Initialize equity history for the warmup period (flat)
        # This solves the "starts at day 86" confusion
        for i in range(start_idx):
            row = self.data.iloc[i]
            d = self._format_date(row.name)
            self.equity_series.append({'time': d, 'value': self.capital})
        
        for i in range(start_idx, len(self.data)): 
            row = self.data.iloc[i]
            current_date = self._format_date(row.name)
            current_price = row['Close']
            
            # 1. Manage Open Position
            if self.position:
                self._check_exit(row, current_date)
                # Mark to market
                eq = self._current_equity(current_price)
                self.equity_curve.append(eq)
                self.equity_series.append({'time': current_date, 'value': eq})
                continue 
            
            # 2. Check Entry
            signal = row.get('ai_signal', 'HOLD')
            confidence = row.get('ai_conf', 50)
            
            if signal == "BUY":
                # Call Risk Engine
                atr = row.get('atr_14', current_price*0.02)
                vol_type = "High" if row.get('vol_10', 0) > 0.02 else "Low"
                
                # Map numeric confidence to Level
                conf_level = "Medium"
                if confidence >= 70: conf_level = "High"
                elif confidence < 40: conf_level = "Low"
                
                risk_params = self.risk_engine.calculate_trade_params(
                    entry_price=current_price,
                    atr=atr,
                    confidence_level=conf_level,
                    volatility_type=vol_type,
                    horizon="Swing"
                )
                
                if risk_params['allowed']:
                    self.position = {
                        'entry': current_price,
                        'size': risk_params['position_size'],
                        'sl': risk_params['stop_loss'],
                        'tp': risk_params['take_profit'],
                        'entry_time': current_date # Store formatted date string
                    }
                    print(f"[{current_date}] OPEN BUY @ {current_price:.2f} | Conf: {confidence:.0f}% | Size: {self.position['size']:.2f}")

            eq = self._current_equity(current_price)
            self.equity_curve.append(eq)
            self.equity_series.append({'time': current_date, 'value': eq})
            
        # Cleanup: Force Close Open Position at the end
        if self.position:
            last_row = self.data.iloc[-1]
            last_date = self._format_date(last_row.name)
            exit_price = last_row['Close']
            commission = 2.0
            pnl_gross = (exit_price - self.position['entry']) * self.position['size']
            pnl_net = pnl_gross - commission
            self.capital += pnl_net
            self.trades.append({
                'pnl': pnl_net,
                'return_pct': (exit_price - self.position['entry']) / self.position['entry'],
                'reason': "End of Backtest",
                'entry_date': self.position['entry_time'],
                'exit_date': last_date,
                'entry_price': self.position['entry'],
                'exit_price': exit_price,
                'size': self.position['size'], # Capture Size
                'market_regime': self._determine_regime(last_row)
            })
            print(f"[{last_date}] FORCED CLOSE @ {exit_price:.2f} | PnL: {pnl_net:.2f}")
            self.position = None
            
        return self._generate_report()

    def _generate_report(self):
        # We pass the raw float values to metrics engine
        report = self.metrics_engine.calculate(self.trades, self.equity_curve)
        
        # Overwrite the simple equity curve with our date-aware series for the UI
        # We'll call it 'equity_curve' in the output report because that's what UI expects
        report['equity_curve'] = self.equity_series 
        
        # Add Regime Analysis
        regime_stats = {}
        for trade in self.trades:
            reg = trade.get('market_regime', 'Unknown')
            if reg not in regime_stats: regime_stats[reg] = {'wins': 0, 'losses': 0, 'total': 0, 'pnl': 0.0}
            
            s = regime_stats[reg]
            s['total'] += 1
            s['pnl'] += trade['pnl']
            if trade['pnl'] > 0: s['wins'] += 1
            else: s['losses'] += 1
            
        report['regime_analysis'] = regime_stats
        return report

    def _check_exit(self, row, current_date):
        low = row['Low']
        high = row['High']
        
        sl = self.position['sl']
        tp = self.position['tp']
        
        exit_price = None
        exit_reason = None
        
        if low <= sl:
            exit_price = sl 
            exit_reason = "Stop Loss"
        elif high >= tp:
            exit_price = tp
            exit_reason = "Take Profit"
            
        if exit_price:
            slippage = exit_price * 0.0005
            commission = 2.0
            
            effective_exit = exit_price - slippage
            
            pnl_gross = (effective_exit - self.position['entry']) * self.position['size']
            pnl_net = pnl_gross - commission
            
            self.capital += pnl_net
            self.trades.append({
                'pnl': pnl_net,
                'return_pct': (effective_exit - self.position['entry']) / self.position['entry'],
                'reason': exit_reason,
                'entry_date': self.position['entry_time'],
                'exit_date': current_date,
                'entry_price': self.position['entry'],
                'exit_price': effective_exit,
                'size': self.position['size'],  # Capture Size
                'market_regime': self._determine_regime(row)
            })
            print(f"[{current_date}] CLOSE {exit_reason} @ {exit_price:.2f} | PnL: {pnl_net:.2f}")
            self.position = None

    def _determine_regime(self, row):
        # Determine regime based on SMA / Volatility
        sma = row.get('sma_50', 0)
        close = row['Close']
        vol = row.get('vol_10', 0)
        
        if vol > 0.02: return "High Volatility"
        if close > sma: return "Bull"
        return "Bear"

    def _current_equity(self, price):
        if self.position:
            unrealized = (price - self.position['entry']) * self.position['size']
            return self.capital + unrealized
        return self.capital

    def _precompute_ai_signals(self):
        """
        Batch process the entire dataframe through models to speed up backtest.
        """
        print("Pre-computing AI signals...")
        
        # Defaults
        self.data['ai_signal'] = 'HOLD'
        self.data['ai_conf'] = 50.0
        
        if 'quant_daily' not in self.models:
             print("Warning: Quant model not found for backtest")
             return

        # 1. Batch Quant Prediction
        try:
            features = self.features_list.get('quant_daily', [])
            # Fill missing
            input_df = self.data.copy()
            for col in features:
                if col not in input_df.columns: input_df[col] = 0.0
            
            input_scaled = self.scalers['quant_daily'].transform(input_df[features])
            probs = self.models['quant_daily'].predict_proba(input_scaled) # (N, 3)
            
            # 0=Sell, 1=Hold, 2=Buy
            self.data['prob_sell'] = probs[:, 0]
            self.data['prob_hold'] = probs[:, 1]
            self.data['prob_buy'] = probs[:, 2]
            
        except Exception as e:
            print(f"Batch Quant Error: {e}")
            self.data['prob_buy'] = 0.0
            self.data['prob_sell'] = 0.0

        # 2. LSTM Prediction (Simulated for speed if missing)
        self.data['lstm_pred'] = np.where(self.data['Close'] > self.data['sma_50'], 0.01, -0.01)
        
        # 3. Apply Decision Logic
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            
            # Logic Patch: Alias Intraday to Daily
            d_buy = row.get('prob_buy', 0)
            d_sell = row.get('prob_sell', 0)
            
            daily_probs = {'buy': d_buy, 'sell': d_sell, 'hold': row.get('prob_hold', 0)}
            intraday_probs = daily_probs 
            
            # Technicals
            technicals = {
                "rsi_14": float(row.get('rsi_14', 50)),
                "adx_14": float(row.get('adx_14', 20)),
                "bb_width": float(row.get('bb_width', 0.1)),
                "macd": float(row.get('macd', 0)),
            }
            vol_type = "High" if row.get('vol_10', 0) > 0.02 else "Medium"
            
            decision = self.decision_engine.evaluate(
                ticker="BACKTEST",
                daily_probs=daily_probs,
                intraday_probs=intraday_probs,
                scalper_signal="Neutral",
                lstm_pred=float(row.get('lstm_pred', 0)),
                regression_pred=0.0,
                volatility_type=vol_type,
                technicals=technicals,
                sentiment_score=0.5,
                macro_trend="Neutral",
                threshold=0.25 
            )
            
            self.data.at[self.data.index[i], 'ai_signal'] = decision['final_action']
            self.data.at[self.data.index[i], 'ai_conf'] = decision['confidence_score']
            
            # DEBUG
            if d_buy > 0.3:
                print(f"[{row.name}] P_Buy: {d_buy:.2f} | Action: {decision['final_action']} | Conf: {decision['confidence_score']}")
            
        print("AI Signals pre-computed.")

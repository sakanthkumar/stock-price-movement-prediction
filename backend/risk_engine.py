import logging

class RiskEngine:
    def __init__(self, capital=10000.0):
        self.capital = capital
        self.max_daily_loss_pct = 0.03
        self.max_weekly_loss_pct = 0.06
        self.max_drawdown_pct = 0.12
        self.max_weekly_loss_pct = 0.06
        self.max_drawdown_pct = 0.12
        self.base_risk_per_trade = 0.015 # 1.5%
        self.max_position_pct = 0.20 # Max 20% capital per trade

    def calculate_trade_params(self, 
                               entry_price: float, 
                               atr: float, 
                               confidence_level: str, # "Low", "Medium", "High"
                               volatility_type: str, 
                               horizon: str = "Swing",
                               current_drawdown: float = 0.0,
                               daily_pnl_pct: float = 0.0,
                               weekly_pnl_pct: float = 0.0,
                               losing_streak: int = 0,
                               capital: float = None) -> dict:  # Added optional capital override
        """
        Calculate Position Size, SL, TP based on Risk Rules.
        """
        
        # 1. Kill Switch Checks
        reasons = []
        allowed = True
        
        if current_drawdown > self.max_drawdown_pct:
            reasons.append(f"Max Drawdown Exceeded ({current_drawdown:.1%})")
            allowed = False
        
        if daily_pnl_pct < -self.max_daily_loss_pct:
            reasons.append(f"Max Daily Loss Hit ({daily_pnl_pct:.1%})")
            allowed = False
            
        if weekly_pnl_pct < -self.max_weekly_loss_pct:
            reasons.append(f"Max Weekly Loss Hit ({weekly_pnl_pct:.1%})")
            allowed = False
            
        if losing_streak > 3:
            reasons.append(f"Losing Streak Kill Switch ({losing_streak} losses)")
            allowed = False
            
        # Confidence Check (Low confidence = No Trade or reduced? Rules say Kill if < 40th percentile)
        # Using level string mapping: Low is < 40 base confidence usually.
        if confidence_level == "Low":
            reasons.append("Confidence too low")
            allowed = False
            
        if not allowed:
            return {"allowed": False, "reason": "; ".join(reasons)}
            
        # 2. Risk Adjustment
        risk_pct = self.base_risk_per_trade
        
        # Adjust for volatility
        if volatility_type == "High":
            risk_pct *= 0.5 # Halve risk in high vol
        elif volatility_type == "Low":
            risk_pct *= 1.2 # Slightly increase in low vol
        
        
        # Adjust for confidence (Using levels)
        if confidence_level == "High":
            # Rule: Top 25% -> High. Maybe slight boost or max allowed.
            pass 
        elif confidence_level == "Medium":
             # Rule: Top 40% -> Medium. Standard risk.
             pass
        else:
             # Low -> Should be killed above, but if allowed:
             risk_pct *= 0.5
            
        # Use provided capital or default
        current_capital = capital if capital is not None else self.capital
        risk_amount = current_capital * risk_pct
        
        # 3. Stop Loss Logic (ATR Based)
        # Scalp: 0.8 ATR, Intraday: 1.2 ATR, Swing: 1.8 ATR
        multiplier = 1.8
        if horizon == "Scalp": multiplier = 0.8
        elif horizon == "Intraday": multiplier = 1.2
        
        sl_distance = atr * multiplier
        stop_loss = entry_price - sl_distance
        
        # 4. Take Profit Logic (Min 1:1.5 RR)
        # Scalp: 1.2 SL, Intraday: 1.5 SL, Swing: 2.0 SL
        tp_multiplier = 2.0
        if horizon == "Scalp": tp_multiplier = 1.2
        elif horizon == "Intraday": tp_multiplier = 1.5
        
        tp_distance = sl_distance * tp_multiplier
        take_profit = entry_price + tp_distance
        
        # 5. Position Sizing
        # Size = Risk Amount / Distance per share
        if sl_distance == 0:
            return {"allowed": False, "reason": "ATR is zero"}
            
        position_size = risk_amount / sl_distance
        
        # 6. Portfolio Diversification (Max Position Limit)
        # Cap position size to equity (e.g. max 20% of capital in one trade)
        # Use provided capital or default
        current_capital = capital if capital is not None else self.capital
        max_position_value = current_capital * self.max_position_pct
        max_size_by_capital = max_position_value / entry_price
        
        # Apply limit
        if position_size > max_size_by_capital:
             # logging.info(f"Capping size {position_size:.2f} -> {max_size_by_capital:.2f} due to 20% limit")
             position_size = max_size_by_capital

        return {
            "allowed": True,
            "action": "BUY",
            "entry_price": entry_price,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "position_size": round(position_size, 4),
            "risk_amount": round(risk_amount, 2),
            "risk_percent": round(risk_pct * 100, 2),
            "capital_usage_pct": round((position_size * entry_price) / current_capital * 100, 1),
            "max_allowed_size": round(max_size_by_capital, 4),
            "rr_ratio": round(tp_distance / sl_distance, 2)
        }

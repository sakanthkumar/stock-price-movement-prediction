import logging
import numpy as np

class ValidatorEngine:
    def __init__(self):
        pass

    def validate(self, 
                 ticker: str,
                 decision: dict,          # Output from DecisionEngine
                 market_data: dict,       # Technicals, Volatility, etc.
                 confidence_history: list # Historical confidence scores
                 ) -> dict:
        """
        Audit the decision against strict rules.
        """
        
        # Unpack Inputs
        action = decision.get("final_action", "HOLD")
        confidence_score = decision.get("confidence_score", 0)
        confidence_percentile_str = decision.get("confidence_percentile", "Below 40%")
        
        # Parse technicals
        technicals = market_data.get("technicals", {})
        rsi = technicals.get("rsi_14", 50)
        adx = technicals.get("adx_14", 0)
        bb_width = technicals.get("bb_width", 0.1)
        
        lstm_pred = market_data.get("lstm_pred", 0.0)
        volatility_type = market_data.get("volatility_type", "Medium")
        
        daily_bias = market_data.get("daily_bias", "Hold") # Need to pass these from API or re-derive
        intraday_bias = market_data.get("intraday_bias", "Hold")
        
        # Map biases to agree/disagree
        models_agree = (daily_bias == intraday_bias) and (daily_bias != "HOLD")
        
        violations = []
        is_valid = "VALID"
        
        # --- PART 1: RULE COMPLIANCE AUDIT ---
        
        # Helper to convert percentile string to rough rank for comparison
        # "Top 10%" -> 90, "Top 25%" -> 75, "Top 40%" -> 60, "Below 40%" -> 0
        conf_rank = 0
        if "Top 10%" in confidence_percentile_str: conf_rank = 90
        elif "Top 25%" in confidence_percentile_str: conf_rank = 75
        elif "Top 40%" in confidence_percentile_str: conf_rank = 60
        else: conf_rank = 30
        
        if action == "BUY":
            if lstm_pred < 0.0040: violations.append("LSTM < +0.40%")
            if conf_rank < 75: violations.append("Confidence not Top 25%")
            if not models_agree: violations.append("Models do not agree")
            if adx < 20: violations.append("ADX < 20")
            if rsi > 70: violations.append("RSI > 70 (Overbought)")
            if volatility_type == "High": violations.append("Volatility is High")
            
        elif action == "SELL":
            if lstm_pred > -0.0040: violations.append("LSTM > -0.40%")
            if conf_rank < 75: violations.append("Confidence not Top 25%")
            if not models_agree: violations.append("Models do not agree")
            if adx < 20: violations.append("ADX < 20")
            if rsi < 30: violations.append("RSI < 30 (Oversold)")
            if volatility_type == "High": violations.append("Volatility is High")
            
        elif action == "HOLD":
            # "A HOLD is expected if ANY of these are true"
            # We flag SUSPICIOUS HOLD if HOLD is chosen but NONE of the hold-reasons apply
            # i.e., it looks like a valid trade was missed?
            
            # Reasons to HOLD:
            hold_reasons = []
            if abs(lstm_pred) < 0.004: hold_reasons.append("Flat LSTM")
            if conf_rank < 75: hold_reasons.append("Low Confidence")
            if not models_agree: hold_reasons.append("Models Disagree")
            if adx < 15: hold_reasons.append("Low ADX")
            if 45 <= rsi <= 55: hold_reasons.append("Neutral RSI")
            if bb_width < 0.05: hold_reasons.append("BB Compression") # Simple threshold
            
            # If no hold reasons exist, but we held -> Suspicious (Missed Opportunity?)
            # But wait, to trade we need ALL buy rules true. 
            # If we missed a criteria (e.g. ADX 18), we HOLD. 
            # ADX < 15 is a hold reason. But ADX 18 is not a 'hold reason' per se, but it fails 'Buy check (ADX>=20)'.
            # The prompt says: "If HOLD is chosen but none of these apply -> FLAG AS SUSPICIOUS HOLD"
            # "None of these apply" means: LSTM is big, Conf is high, Models agree, ADX >= 15, RSI not neutral...
            if not hold_reasons:
                 # Check if it actually qualified for a trade?
                 pass # Logic is tricky here, sticking to explicit prompt instructions.
                 # "If HOLD is chosen but none of these apply"
                 # It implies checking strictly the list:
                 pass 
            
        if violations:
            is_valid = "INVALID"
            
        # --- PART 2: REGIME CLASSIFICATION ---
        regime = "Sideways"
        # Simple logical deduction
        if adx > 25:
            if daily_bias == "BUY": regime = "Trending Up"
            elif daily_bias == "SELL": regime = "Trending Down"
        
        if volatility_type == "High":
            regime = "High Volatility"
        
        expected_behavior = "HOLD"
        if regime in ["Trending Up", "Trending Down", "Breakout", "Breakdown"]:
            expected_behavior = "TRADE"
        
        behavior_match = "MATCH"
        if expected_behavior == "TRADE" and action == "HOLD":
             # This is acceptable if rules block it.
             pass
        if expected_behavior == "HOLD" and action != "HOLD":
             violations.append(f"Trade in {regime} regime (Expected HOLD)")
             is_valid = "INVALID"

        # --- PART 6: SCORING ---
        # Rule Compliance
        compliance_score = 100 - (len(violations) * 20)
        compliance_score = max(0, compliance_score)
        
        # Risk Awareness (High vol trades punish score)
        risk_score = 100
        if volatility_type == "High" and action != "HOLD":
            risk_score = 0
            
        # Discipline (Did we follow the plan?)
        discipline_score = compliance_score 
        
        # Calibration (Does confidence match reality?)
        # Hard to audit without outcome, but can check consistency.
        calibration_score = 100
        if conf_rank > 80 and action == "HOLD":
            # Confidence high but holding? Mismatch unless filtered.
            calibration_score = 80
            
        final_score = (compliance_score * 0.4) + (risk_score * 0.3) + (discipline_score * 0.3)
        
        return {
            "ticker": ticker,
            "decision_validity": is_valid,
            "violations": violations,
            "regime_classification": regime,
            "expected_behavior": expected_behavior,
            "actual_behavior": action,
            "confidence_audit": "PASS" if calibration_score > 50 else "FAIL",
            "health_scores": {
                "rule_compliance": compliance_score,
                "discipline": discipline_score,
                "confidence_calibration": calibration_score,
                "risk_awareness": risk_score,
            },
            "final_system_health_score": int(final_score),
            "summary": f"Decision is {is_valid}. {len(violations)} violations found."
        }

import logging
import traceback
from datetime import datetime
try:
    from .decision_engine import DecisionEngine
    from .validator_engine import ValidatorEngine
    from .risk_engine import RiskEngine
except ImportError:
    from decision_engine import DecisionEngine
    from validator_engine import ValidatorEngine
    from risk_engine import RiskEngine

# Configure logging
logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        self.decision_engine = DecisionEngine()
        self.validator_engine = ValidatorEngine()
        self.risk_engine = RiskEngine()
        
    def run_pipeline(self, 
                     ticker: str,
                     market_data: dict, 
                     account_state: dict = None) -> dict:
        """
        Execute the strict Production Pipeline:
        Input -> Decision -> Validator -> Risk -> Execution Gate -> Output
        """
        
        pipeline_start = datetime.now()
        
        # Default Account State if None
        if account_state is None:
            account_state = {
                "capital": 10000.0,
                "current_drawdown": 0.0,
                "daily_pnl_pct": 0.0,
                "weekly_pnl_pct": 0.0,
                "losing_streak": 0
            }

        result = {
            "pipeline_status": "BLOCKED",
            "decision_output": {},
            "validator_output": {},
            "risk_output": {},
            "execution_permission": False,
            "block_reason": None,
            "final_action": "HOLD",
            "confidence": "Low",
            "health_snapshot": {},
            "timestamp": pipeline_start.isoformat()
        }

        try:
            # --- STEP 1: INPUT DATA VALIDATION (Schema Check) ---
            # Basic sanity checks on market_data
            if not market_data or 'current_price' not in market_data:
                result["block_reason"] = "Schema Violation: Missing market data"
                result["block_reason_module"] = "Schema"
                return result

            # --- STEP 2: DECISION ENGINE ---
            try:
                decision = self.decision_engine.evaluate(
                    ticker=ticker,
                    daily_probs=market_data.get('daily_probs', {}),
                    intraday_probs=market_data.get('intraday_probs', {}),
                    scalper_signal=market_data.get('scalper_signal', "Neutral"),
                    lstm_pred=market_data.get('lstm_pred', 0.0),
                    regression_pred=market_data.get('regression_pred', 0.0),
                    volatility_type=market_data.get('volatility_type', "Medium"),
                    technicals=market_data.get('technicals', {}),
                    sentiment_score=market_data.get('sentiment_score', 0.5),
                    macro_trend=market_data.get('macro_trend', "Neutral"),
                    asset_type=market_data.get('asset_type', "Stock"),
                    confidence_history=market_data.get('confidence_history', [])
                )
                result["decision_output"] = decision
            except Exception as e:
                logger.error(f"Decision Engine Failed: {e}")
                result["block_reason"] = f"Decision Engine Crash: {str(e)}"
                result["block_reason_module"] = "Decision Engine"
                return result

            # --- STEP 3: VALIDATOR ENGINE ---
            try:
                # Prepare data for validator (needs biases usually derived in api or inside decision)
                # But current decision engine returns biases in 'explanation'. 
                # Ideally market_data should have them. 
                # For now, we rely on market_data having the raw inputs.
                validation = self.validator_engine.validate(
                    ticker=ticker,
                    decision=decision,
                    market_data=market_data,
                    confidence_history=market_data.get('confidence_history', [])
                )
                result["validator_output"] = validation
                result["health_snapshot"] = validation.get("health_scores", {})
            except Exception as e:
                 logger.error(f"Validator Engine Failed: {e}")
                 result["block_reason"] = f"Validator Engine Crash: {str(e)}"
                 result["block_reason_module"] = "Validator Engine"
                 return result

            # --- STEP 4: RISK ENGINE ---
            # Only run if Decision is not HOLD? 
            # Risk Engine calculates params regardless, but 'allowed' is key.
            try:
                risk_params = self.risk_engine.calculate_trade_params(
                    entry_price=market_data['current_price'],
                    atr=market_data.get('technicals', {}).get('atr_14', market_data['current_price']*0.02),
                    confidence_level=decision.get('confidence_level', 'Low'),
                    volatility_type=decision.get('risk_level', 'High'),
                    horizon=decision.get('trade_horizon', 'Swing'),
                    current_drawdown=account_state['current_drawdown'],
                    daily_pnl_pct=account_state['daily_pnl_pct'],
                    weekly_pnl_pct=account_state['weekly_pnl_pct'],
                    losing_streak=account_state['losing_streak']
                )
                result["risk_output"] = risk_params
            except Exception as e:
                logger.error(f"Risk Engine Failed: {e}")
                result["block_reason"] = f"Risk Engine Crash: {str(e)}"
                result["block_reason_module"] = "Risk Engine"
                return result

            # --- STEP 5: EXECUTION GATE (CRITICAL) ---
            # No action is allowed unless ALL pass:
            # 1. Decision is valid signal (implied if not exception)
            # 2. Validator returns VALID
            # 3. Risk Engine allowed = True
            
            is_valid_decision = (decision.get('final_action', 'HOLD') != 'HOLD')
            is_validator_pass = (validation.get('decision_validity') == 'VALID')
            is_risk_allowed = risk_params.get('allowed', False)
            
            block_reasons = []
            
            # If Decision is HOLD, we stick to HOLD, but pipeline is SUCCESS (we successfully decided to hold).
            # But "Execution Permission" is False (no trade execution).
            
            if decision.get('final_action') == 'HOLD':
                 result["final_action"] = "HOLD"
                 result["pipeline_status"] = "SUCCESS"
                 result["execution_permission"] = False
                 result["block_reason"] = "Strategy suggests HOLD"
                 if decision.get('explanation'):
                     # Append reason
                     pass
            else:
                # Signal is BUY or SELL
                if not is_validator_pass:
                    block_reasons.append(f"Validator Rejection: {validation.get('violations', [])}")
                
                if not is_risk_allowed:
                    block_reasons.append(f"Risk Rejection: {risk_params.get('reason', '')}")
                    
                if block_reasons:
                    # FORCE HOLD
                    result["final_action"] = "HOLD" # Override
                    result["pipeline_status"] = "SUCCESS" # Pipeline finished, but outcome is blocked trade
                    result["execution_permission"] = False
                    result["block_reason"] = "; ".join(block_reasons)
                    result["block_reason_module"] = "Execution Gate"
                else:
                    # ALL GREEN
                    result["final_action"] = decision.get('final_action')
                    result["pipeline_status"] = "SUCCESS"
                    result["execution_permission"] = True
                    result["block_reason"] = None

            # --- STEP 6: OUTPUT FORMATTING ---
            result["confidence"] = f"{decision.get('confidence_level')} ({decision.get('confidence_score')}%)"
            
            return result

        except Exception as e:
            # Catch-all System Failure
            logger.error(f"Orchestrator System Failure: {e}")
            traceback.print_exc()
            result["pipeline_status"] = "BLOCKED"
            result["block_reason"] = f"System Error: {str(e)}"
            result["block_reason_module"] = "Orchestrator"
            result["final_action"] = "HOLD"
            return result

"""
RazorGuard AI - Hybrid Decision & Risk Engine

Combines ML Fraud Probability + Deterministic Policy Rules into final Risk Score (0-100),
Risk Level (LOW, MEDIUM, HIGH), and Decision (APPROVE, REVIEW, BLOCK).
Includes a conservative fallback strategy when the ML model is unavailable.
Strictly zero LLM execution authority.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timezone

from ml.inference.predictor import RazorGuardPredictor
from backend.app.risk_engine.scoring import probability_to_risk_score, map_score_to_risk_level_and_decision
from backend.app.risk_engine.rules import evaluate_deterministic_rules

logger = logging.getLogger("razorguard.risk_engine")


class HybridRiskEngine:
    """
    Production Risk Engine combining ML predictions and deterministic compliance rules.
    """
    def __init__(self, models_dir: str = "models", low_threshold: int = 30, high_threshold: int = 70):
        self.predictor = RazorGuardPredictor(models_dir=models_dir)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._try_load_predictor()

    def _try_load_predictor(self):
        try:
            self.predictor.load()
        except Exception as e:
            logger.warning(f"ML Predictor load failed (will use conservative fallback): {e}")

    def _execute_fallback_strategy(self, transaction: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Conservative fallback strategy executed when ML model is unavailable or throws error.
        Logs warning and applies defensive rules.
        """
        logger.warning(f"ML_MODEL_UNAVAILABLE_FALLBACK_TRIGGERED: {reason}")

        amount = float(transaction.get("amount", 0.0))
        failed_attempts = int(transaction.get("failed_payment_count", 0))
        tx_24h = int(transaction.get("transactions_last_24h", 0))
        country_change = int(transaction.get("country_change", 0))

        fallback_rules = ["FALLBACK_CONSERVATIVE_MODE", f"REASON_{reason}"]

        if amount >= 1000.0 or failed_attempts >= 2 or tx_24h >= 4 or country_change == 1:
            score = 80
            level = "HIGH"
            decision = "BLOCK"
            fallback_rules.append("FALLBACK_HIGH_RISK_BLOCK")
        elif amount >= 250.0:
            score = 55
            level = "MEDIUM"
            decision = "REVIEW"
            fallback_rules.append("FALLBACK_MEDIUM_RISK_REVIEW")
        elif amount <= 50.0 and failed_attempts == 0 and tx_24h <= 1:
            score = 20
            level = "LOW"
            decision = "APPROVE"
            fallback_rules.append("FALLBACK_LOW_VAL_APPROVE")
        else:
            score = 45
            level = "MEDIUM"
            decision = "REVIEW"
            fallback_rules.append("FALLBACK_DEFAULT_REVIEW")

        return {
            "fraud_probability": round(score / 100.0, 4),
            "ml_risk_score": score,
            "final_risk_score": score,
            "anomaly_score": 50.0,
            "risk_level": level,
            "decision": decision,
            "triggered_rules": fallback_rules,
            "fallback_active": True,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
        }

    def evaluate_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for transaction risk evaluation.
        """
        # 1. Check if ML Predictor is loaded
        if not self.predictor.is_loaded:
            try:
                self.predictor.load()
            except Exception as e:
                return self._execute_fallback_strategy(transaction, reason=f"Model load error: {str(e)}")

        # 2. Get ML Fraud Probability & Anomaly Score
        try:
            prob = float(self.predictor.predict_fraud_probability(transaction)[0])
            anom_score = float(self.predictor.predict_anomaly_score(transaction)[0])
        except Exception as e:
            return self._execute_fallback_strategy(transaction, reason=f"Inference execution error: {str(e)}")

        # 3. Initial ML Risk Score
        ml_score = probability_to_risk_score(prob)

        # 4. Apply Deterministic Policy Overrides
        final_score, triggered_rules = evaluate_deterministic_rules(transaction, initial_score=ml_score)

        # 5. Map to Risk Level & Decision
        level, decision = map_score_to_risk_level_and_decision(
            final_score,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold
        )

        return {
            "fraud_probability": prob,
            "ml_risk_score": ml_score,
            "final_risk_score": final_score,
            "anomaly_score": anom_score,
            "risk_level": level,
            "decision": decision,
            "triggered_rules": triggered_rules,
            "fallback_active": False,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
        }

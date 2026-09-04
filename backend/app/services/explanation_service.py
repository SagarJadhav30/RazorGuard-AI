"""
RazorGuard AI - Full Unified Explanation Service

Integrates Hybrid Risk Engine + SHAP Feature Attribution + LLM Risk Analyst Layer.
Provides 100% feature-grounded, human-readable explanations with zero LLM decision authority.
"""

import pandas as pd
import logging
from typing import Dict, Any, List

from backend.app.risk_engine.engine import HybridRiskEngine
from ml.inference.shap_explainer import RazorGuardSHAPExplainer
from backend.app.services.llm_engine import LLMRiskAnalyst

logger = logging.getLogger("razorguard.explanation_service")


class ExplanationService:
    """
    Unified Explainable AI Service for RazorGuard AI.
    """
    def __init__(self, models_dir: str = "models"):
        self.risk_engine = HybridRiskEngine(models_dir=models_dir)
        self.shap_explainer = RazorGuardSHAPExplainer(models_dir=models_dir)
        self.llm_analyst = LLMRiskAnalyst()

    def explain_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes Risk Engine evaluation, SHAP feature attribution, and LLM analyst narration.
        Returns complete structured JSON payload.
        """
        # 1. Hybrid Risk Engine Evaluation (ML + Policy Rules)
        risk_result = self.risk_engine.evaluate_transaction(transaction)

        # 2. SHAP Feature Attribution
        df_raw = pd.DataFrame([transaction])
        try:
            shap_result = self.shap_explainer.explain_single_transaction(df_raw, top_k=5)
        except Exception as e:
            shap_result = {
                "risk_factors": [],
                "protective_factors": [],
                "error": str(e)
            }

        # 3. Clean Factor Formatting
        clean_risk_factors = []
        for rf in shap_result.get("risk_factors", []):
            clean_rf = {k: v for k, v in rf.items() if k != "abs_impact"}
            clean_risk_factors.append(clean_rf)

        clean_protective_factors = []
        for pf in shap_result.get("protective_factors", []):
            clean_pf = {k: v for k, v in pf.items() if k != "abs_impact"}
            clean_protective_factors.append(clean_pf)

        # 4. Build Evidence Object
        evidence = {
            "fraud_probability": risk_result["fraud_probability"],
            "ml_risk_score": risk_result["ml_risk_score"],
            "final_risk_score": risk_result["final_risk_score"],
            "risk_level": risk_result["risk_level"],
            "decision": risk_result["decision"],
            "anomaly_score": risk_result["anomaly_score"],
            "triggered_rules": risk_result["triggered_rules"],
            "fallback_active": risk_result.get("fallback_active", False),
            "risk_factors": clean_risk_factors,
            "protective_factors": clean_protective_factors,
            "evaluation_timestamp": risk_result["evaluation_timestamp"]
        }

        # 5. Generate LLM Risk Analyst Narration
        try:
            llm_narration = self.llm_analyst.generate_analyst_explanation(evidence, raw_transaction=transaction)
        except Exception:
            logger.exception("LLM explanation unavailable; using deterministic fallback")
            llm_narration = self.llm_analyst.fallback.generate_explanation(evidence)

        # 6. Merge Evidence & Narration
        explanation_payload = {
            **evidence,
            "llm_explanation": {
                "summary_headline": llm_narration.get("summary_headline", ""),
                "why_flagged": llm_narration.get("why_flagged", ""),
                "top_risk_factors": llm_narration.get("top_risk_factors", []),
                "recommended_analyst_action": llm_narration.get("recommended_analyst_action", ""),
                "confidence_and_limitations": llm_narration.get("confidence_and_limitations", ""),
                "disclaimer": llm_narration.get("disclaimer", ""),
                "provider_used": llm_narration.get("provider_used", "offline_deterministic_template")
            }
        }

        return explanation_payload

"""
RazorGuard AI - Phase 7 SHAP Explainability Unit Tests
"""

import os
import pytest
import pandas as pd

from ml.utils.data_generator import get_preset_scenarios
from ml.inference.shap_explainer import RazorGuardSHAPExplainer
from backend.app.services.explanation_service import ExplanationService


def test_shap_explainer_initialization():
    """Verify SHAP explainer initializes and computes attributions."""
    explainer = RazorGuardSHAPExplainer(models_dir="models").load()
    presets = get_preset_scenarios()
    df_raw = pd.DataFrame([presets["stolen_card_attack"]["data"]])

    shap_vals, X_proc = explainer.compute_shap_attributions(df_raw)
    assert shap_vals.shape[0] == 1
    assert shap_vals.shape[1] == len(explainer.feature_names)


def test_explanation_service_json_schema():
    """Verify ExplanationService produces valid structured JSON schema."""
    service = ExplanationService(models_dir="models")
    presets = get_preset_scenarios()

    stolen_tx = presets["stolen_card_attack"]["data"]
    exp = service.explain_transaction(stolen_tx)

    # Required JSON schema keys
    assert "fraud_probability" in exp
    assert "final_risk_score" in exp
    assert "risk_level" in exp
    assert "decision" in exp
    assert "risk_factors" in exp
    assert "protective_factors" in exp
    assert "triggered_rules" in exp

    # Check risk factor fields
    if exp["risk_factors"]:
        rf = exp["risk_factors"][0]
        assert "feature" in rf
        assert "feature_name_human" in rf
        assert "shap_value" in rf
        assert "impact_direction" in rf
        assert rf["impact_direction"] == "risk_increasing"
        assert rf["shap_value"] > 0


def test_explanation_groundedness():
    """Verify protective factors contain negative SHAP values."""
    service = ExplanationService(models_dir="models")
    presets = get_preset_scenarios()

    grocery_tx = presets["standard_grocery_inperson"]["data"]
    exp = service.explain_transaction(grocery_tx)

    if exp["protective_factors"]:
        pf = exp["protective_factors"][0]
        assert pf["impact_direction"] == "risk_decreasing"
        assert pf["shap_value"] <= 0

"""
RazorGuard AI - Phase 6 Risk Scoring & Hybrid Decision Engine Unit Tests
"""

import os
import pytest
import pandas as pd

from backend.app.risk_engine.scoring import probability_to_risk_score, map_score_to_risk_level_and_decision
from backend.app.risk_engine.rules import evaluate_deterministic_rules
from backend.app.risk_engine.engine import HybridRiskEngine
from ml.utils.data_generator import get_preset_scenarios


def test_probability_to_score_mapping():
    """Verify conversion of probabilities to integer 0-100 risk scores."""
    assert probability_to_risk_score(0.0) == 0
    assert probability_to_risk_score(0.294) == 29
    assert probability_to_risk_score(0.30) == 30
    assert probability_to_risk_score(0.699) == 70
    assert probability_to_risk_score(1.0) == 100


def test_score_boundaries_mapping():
    """Verify exact boundary values map to correct Risk Level and Decision."""
    # Score 29 -> LOW / APPROVE
    lvl, dec = map_score_to_risk_level_and_decision(29, low_threshold=30, high_threshold=70)
    assert lvl == "LOW" and dec == "APPROVE"

    # Score 30 -> MEDIUM / REVIEW
    lvl, dec = map_score_to_risk_level_and_decision(30, low_threshold=30, high_threshold=70)
    assert lvl == "MEDIUM" and dec == "REVIEW"

    # Score 69 -> MEDIUM / REVIEW
    lvl, dec = map_score_to_risk_level_and_decision(69, low_threshold=30, high_threshold=70)
    assert lvl == "MEDIUM" and dec == "REVIEW"

    # Score 70 -> HIGH / BLOCK
    lvl, dec = map_score_to_risk_level_and_decision(70, low_threshold=30, high_threshold=70)
    assert lvl == "HIGH" and dec == "BLOCK"


def test_preset_scenarios_decisions():
    """Verify hybrid risk engine evaluates preset scenarios correctly."""
    engine = HybridRiskEngine(models_dir="models")
    presets = get_preset_scenarios()

    # 1. Stolen Card Attack -> BLOCK
    stolen = presets["stolen_card_attack"]["data"]
    res_stolen = engine.evaluate_transaction(stolen)
    assert res_stolen["decision"] == "BLOCK"
    assert res_stolen["risk_level"] == "HIGH"
    assert res_stolen["final_risk_score"] >= 70

    # 2. In-Person Grocery Store -> APPROVE
    grocery = presets["standard_grocery_inperson"]["data"]
    res_grocery = engine.evaluate_transaction(grocery)
    assert res_grocery["decision"] == "APPROVE"
    assert res_grocery["risk_level"] == "LOW"
    assert res_grocery["final_risk_score"] < 30


def test_sanctioned_country_rule_override():
    """Verify deterministic rule forces instant BLOCK for sanctioned country."""
    engine = HybridRiskEngine(models_dir="models")
    presets = get_preset_scenarios()

    clean_tx = presets["standard_grocery_inperson"]["data"].copy()
    clean_tx["country"] = "XX"  # Sanctioned country

    res = engine.evaluate_transaction(clean_tx)
    assert res["decision"] == "BLOCK"
    assert res["final_risk_score"] == 100
    assert "RULE_SANCTIONED_COUNTRY_BLOCK" in res["triggered_rules"]


def test_missing_features_and_invalid_inputs():
    """Verify risk engine handles partial or invalid payloads without throwing unhandled exceptions."""
    engine = HybridRiskEngine(models_dir="models")

    partial_payload = {"amount": 500.0, "transactions_last_24h": 2}
    res = engine.evaluate_transaction(partial_payload)

    assert "decision" in res
    assert "final_risk_score" in res
    assert 0 <= res["final_risk_score"] <= 100


def test_fallback_strategy_when_model_unavailable(tmp_path):
    """Verify conservative fallback strategy triggers when model directory is invalid."""
    # Point engine to empty directory with no models
    empty_models_dir = str(tmp_path)
    engine = HybridRiskEngine(models_dir=empty_models_dir)

    tx_high_amount = {"amount": 1500.0, "transactions_last_24h": 1}
    res_high = engine.evaluate_transaction(tx_high_amount)

    assert res_high["fallback_active"] is True
    assert res_high["decision"] == "BLOCK"
    assert "FALLBACK_CONSERVATIVE_MODE" in res_high["triggered_rules"]

    tx_low_val = {"amount": 25.0, "transactions_last_24h": 1, "failed_payment_count": 0}
    res_low = engine.evaluate_transaction(tx_low_val)

    assert res_low["fallback_active"] is True
    assert res_low["decision"] == "APPROVE"

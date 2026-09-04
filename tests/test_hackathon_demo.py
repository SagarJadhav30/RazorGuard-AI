"""
RazorGuard AI - Hackathon Demo Automated Integration Test Suite

Tests:
1. Scenario 1 (Normal Returning Customer -> LOW RISK -> APPROVE)
2. Scenario 2 (New Customer + Unusual Amount -> MEDIUM RISK -> REVIEW)
3. Scenario 3 (Multiple Payment Attempts + Device Reuse -> HIGH RISK -> BLOCK)
4. Scenario 4 (Known VIP Customer High-Value -> LOW RISK -> APPROVE)
5. Full explainability fields verification (Input, ML prediction, Risk score, Decision, SHAP, GenAI, Audit event)
6. Demo Reset mechanism verification
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.demo.scenarios import DEMO_SCENARIOS, get_all_scenarios_metadata, get_scenario_by_id

client = TestClient(app)


def test_list_demo_scenarios_metadata():
    """Verify that all 4 demo scenarios are discoverable with rich presentation metadata."""
    response = client.get("/api/v1/demo/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 4
    scenarios = data["scenarios"]
    scenario_ids = [s["id"] for s in scenarios]
    assert "scenario_1" in scenario_ids
    assert "scenario_2" in scenario_ids
    assert "scenario_3" in scenario_ids
    assert "scenario_4" in scenario_ids


def test_scenario_1_normal_customer_approves():
    """SCENARIO 1: Normal returning customer -> LOW RISK -> APPROVE."""
    res = client.post("/api/v1/demo/execute/scenario_1")
    assert res.status_code == 200
    d = res.json()

    assert d["scenario"]["id"] == "scenario_1"
    assert d["assessment"]["risk_level"] == "LOW"
    assert d["assessment"]["decision"] == "APPROVE"
    assert d["assessment"]["final_risk_score"] < 30

    # Explainability checks
    assert "input" in d
    assert "fraud_probability" in d["assessment"]
    assert "shap_factors" in d
    assert len(d["shap_factors"]["protective_factors"]) > 0
    assert "llm_explanation" in d
    assert d["llm_explanation"]["summary_headline"] != ""
    assert "audit_event" in d
    assert d["audit_event"]["event_type"] == "DEMO_SCENARIO_EXECUTED"
    assert d["audit_event"]["action"] == "DEMO_APPROVE"


def test_scenario_2_new_customer_unusual_amount_reviews():
    """SCENARIO 2: New customer + unusual amount -> MEDIUM RISK -> REVIEW."""
    res = client.post("/api/v1/demo/execute/scenario_2")
    assert res.status_code == 200
    d = res.json()

    assert d["scenario"]["id"] == "scenario_2"
    assert d["assessment"]["risk_level"] == "MEDIUM"
    assert d["assessment"]["decision"] == "REVIEW"
    assert 30 <= d["assessment"]["final_risk_score"] < 70

    # Explainability checks
    assert "input" in d
    assert len(d["shap_factors"]["risk_factors"]) > 0
    assert d["audit_event"]["action"] == "DEMO_REVIEW"


def test_scenario_3_brute_force_attack_blocks():
    """SCENARIO 3: Multiple payment attempts + device reuse + velocity -> HIGH RISK -> BLOCK."""
    res = client.post("/api/v1/demo/execute/scenario_3")
    assert res.status_code == 200
    d = res.json()

    assert d["scenario"]["id"] == "scenario_3"
    assert d["assessment"]["risk_level"] == "HIGH"
    assert d["assessment"]["decision"] == "BLOCK"
    assert d["assessment"]["final_risk_score"] >= 70
    assert len(d["assessment"]["triggered_rules"]) > 0

    # Explainability checks
    assert "input" in d
    assert len(d["shap_factors"]["risk_factors"]) > 0
    assert d["audit_event"]["action"] == "DEMO_BLOCK"


def test_scenario_4_high_value_legitimate_customer_approves():
    """
    SCENARIO 4: Known legitimate customer with high-value purchase -> LOW RISK -> APPROVE
    Demonstrates that high ticket size alone does not trigger a block when balanced by customer loyalty.
    """
    res = client.post("/api/v1/demo/execute/scenario_4")
    assert res.status_code == 200
    d = res.json()

    assert d["scenario"]["id"] == "scenario_4"
    assert d["assessment"]["risk_level"] == "LOW"
    assert d["assessment"]["decision"] == "APPROVE"
    assert d["assessment"]["final_risk_score"] < 30
    assert d["input"]["amount"] == 2450.00

    # Verify SHAP shows amount as a risk factor, but strong protective factors (loyalty/history) keep it approved
    risk_factor_names = [f["feature_name_human"] for f in d["shap_factors"]["risk_factors"]]
    protective_factor_names = [f["feature_name_human"] for f in d["shap_factors"]["protective_factors"]]
    assert any("Amount" in name for name in risk_factor_names)
    assert any("Transaction Count" in name or "Average" in name for name in protective_factor_names)


def test_demo_reset_mechanism():
    """Verifies that demo reset purges demo transactions and registers audit reset event."""
    # Execute a scenario first
    client.post("/api/v1/demo/execute/scenario_1")

    # Now execute reset
    reset_res = client.post("/api/v1/demo/reset")
    assert reset_res.status_code == 200
    reset_data = reset_res.json()
    assert reset_data["status"] == "success"
    assert "TXN_DEMO_SCENARIO_1" in reset_data["purged_demo_transactions"]

    # Verify audit log includes reset entry
    audit_res = client.get("/api/v1/audit?limit=10")
    assert audit_res.status_code == 200
    actions = [item["action"] for item in audit_res.json()["items"]]
    assert "DEMO_RESET" in actions

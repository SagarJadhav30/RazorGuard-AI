"""
RazorGuard AI - Hackathon Demo Scenarios & Reset Mechanism Automated Tests

Tests:
1. Scenario 1 (Normal returning customer -> LOW RISK / APPROVE)
2. Scenario 2 (New customer + unusual amount -> MEDIUM RISK / REVIEW)
3. Scenario 3 (Multiple payment attempts + device reuse -> HIGH RISK / BLOCK)
4. Scenario 4 (Known legitimate customer with high-value purchase -> LOW RISK / APPROVE)
5. Demonstration that high amount alone does NOT mean fraud (comparing Scenario 4 vs Scenario 3)
6. Demo reset mechanism clears and re-initializes database cleanly.
"""

import pytest
from backend.app.services.explanation_service import ExplanationService
from backend.app.database.session import SessionLocal
from backend.app.models.db_models import TransactionModel, RiskAssessmentModel, AuditLogModel
from ml.utils.data_generator import get_preset_scenarios
from scripts.live_demo import reset_demo_database, run_demo_scenario


@pytest.fixture
def service():
    return ExplanationService(models_dir="models")


def test_demo_scenario_1_returning_customer(service):
    """Scenario 1: Normal returning customer must evaluate to LOW RISK and APPROVE."""
    presets = get_preset_scenarios()
    scen1 = presets["scenario_1_returning_customer"]

    result = service.explain_transaction(scen1["data"])

    assert result["decision"] == "APPROVE"
    assert result["risk_level"] == "LOW"
    assert result["final_risk_score"] < 30
    assert result["llm_explanation"]["summary_headline"]
    assert "LOW RISK" in result["llm_explanation"]["summary_headline"] or "APPROVED" in result["llm_explanation"]["summary_headline"]


def test_demo_scenario_2_new_customer_unusual_amount(service):
    """Scenario 2: New customer + unusual amount must evaluate to MEDIUM RISK and REVIEW."""
    presets = get_preset_scenarios()
    scen2 = presets["scenario_2_new_customer_unusual_amount"]

    result = service.explain_transaction(scen2["data"])

    assert result["decision"] == "REVIEW"
    assert result["risk_level"] == "MEDIUM"
    assert 30 <= result["final_risk_score"] < 70
    assert result["llm_explanation"]["summary_headline"]
    assert "REVIEW" in result["llm_explanation"]["summary_headline"] or "MEDIUM RISK" in result["llm_explanation"]["summary_headline"]


def test_demo_scenario_3_multi_attempt_device_reuse(service):
    """Scenario 3: Multiple attempts + device reuse must evaluate to HIGH RISK and BLOCK."""
    presets = get_preset_scenarios()
    scen3 = presets["scenario_3_multi_attempt_device_reuse"]

    result = service.explain_transaction(scen3["data"])

    assert result["decision"] == "BLOCK"
    assert result["risk_level"] == "HIGH"
    assert result["final_risk_score"] >= 70
    assert result["fraud_probability"] >= 0.70
    assert result["llm_explanation"]["summary_headline"]
    assert "BLOCK" in result["llm_explanation"]["summary_headline"] or "HIGH RISK" in result["llm_explanation"]["summary_headline"]


def test_demo_scenario_4_high_value_legitimate(service):
    """
    Scenario 4: Known legitimate customer with high-value purchase ($3,200)
    must evaluate to LOW RISK and APPROVE, demonstrating high amount alone is not fraud.
    """
    presets = get_preset_scenarios()
    scen4 = presets["scenario_4_high_value_legitimate"]

    result = service.explain_transaction(scen4["data"])

    assert result["decision"] == "APPROVE"
    assert result["risk_level"] == "LOW"
    assert result["final_risk_score"] < 30

    # Verify SHAP shows historical trust factors protect the customer
    protective_names = [pf["feature_name_human"] for pf in result["protective_factors"]]
    assert any("Historical" in name or "Tenure" in name or "Account" in name or "Device" in name for name in protective_names)


def test_high_amount_comparison_scenario_4_vs_scenario_3(service):
    """
    Direct comparison proving amount alone does not cause fraud classification:
    Scenario 4 ($3,200 from trusted customer) -> APPROVE
    Scenario 3 ($1,850 from attack pattern) -> BLOCK
    """
    presets = get_preset_scenarios()
    res_high_legit = service.explain_transaction(presets["scenario_4_high_value_legitimate"]["data"])
    res_lower_fraud = service.explain_transaction(presets["scenario_3_multi_attempt_device_reuse"]["data"])

    # Despite having a higher amount ($3,200 > $1,850), Scenario 4 is approved while Scenario 3 is blocked
    assert presets["scenario_4_high_value_legitimate"]["data"]["amount"] > presets["scenario_3_multi_attempt_device_reuse"]["data"]["amount"]
    assert res_high_legit["decision"] == "APPROVE"
    assert res_lower_fraud["decision"] == "BLOCK"


def test_demo_reset_mechanism():
    """Verify demo reset mechanism empties transactions and audit logs."""
    reset_demo_database()

    with SessionLocal() as db:
        assert db.query(TransactionModel).count() == 0
        assert db.query(RiskAssessmentModel).count() == 0
        assert db.query(AuditLogModel).count() == 0

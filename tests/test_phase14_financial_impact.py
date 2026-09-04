"""Contract and reproducibility tests for Phase 14 financial impact."""

import json
from pathlib import Path

from backend.ml import trainer

ROOT = Path(__file__).parents[1]
COMPONENT = (ROOT / "frontend" / "src" / "components" / "FinancialImpactPage.jsx").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
METRICS = json.loads((ROOT / "backend" / "ml" / "artifacts" / "metrics.json").read_text(encoding="utf-8"))


def test_phase14_financial_impact_is_wired_to_navigation():
    assert "FinancialImpactPage" in APP
    assert "'impact', 'Financial Impact'" in APP
    assert "<FinancialImpactPage />" in APP


def test_phase14_displays_requested_impact_metrics():
    for label in (
        "False positives",
        "False negatives",
        "Average legitimate value",
        "Average fraudulent value",
        "Estimated false-positive cost",
        "Estimated fraud loss avoided",
        "Estimated net benefit",
    ):
        assert label in COMPONENT


def test_phase14_uses_configurable_assumptions_and_disclaimers():
    assert "falseNegativeCost" in COMPONENT
    assert "falsePositiveCost" in COMPONENT
    assert "Configurable assumptions" in COMPONENT
    assert "Estimated" in COMPONENT
    assert "Simulated" in COMPONENT
    assert "Based on test dataset" in COMPONENT
    assert "not realized merchant savings" in COMPONENT


def test_phase14_has_without_and_with_ai_scenario_comparison():
    assert "Without AI" in COMPONENT
    assert "With RazorGuard" in COMPONENT
    for label in (
        "Transactions reviewed",
        "Fraudulent transactions caught",
        "Legitimate incorrectly blocked",
        "Estimated financial impact",
    ):
        assert label in COMPONENT


def test_phase14_calculations_are_based_on_backend_evaluation_fields():
    assert "metrics.confusion_matrix" in COMPONENT
    assert "metrics.test_fraud_count" in COMPONENT
    assert "metrics.test_legit_count" in COMPONENT
    assert "metrics.cost_analysis" in COMPONENT
    assert "Number(c.false_positives" in COMPONENT
    assert "Number(c.false_negatives" in COMPONENT


def test_phase14_metrics_artifact_contains_reproducible_values():
    confusion = METRICS["confusion_matrix"]
    costs = METRICS["cost_analysis"]
    assert costs["average_legitimate_transaction_value_usd"] >= 0
    assert costs["average_fraudulent_transaction_value_usd"] >= 0
    assert costs["fraudulent_transactions_caught"] == confusion["true_positives"]
    assert costs["legitimate_transactions_incorrectly_blocked"] == confusion["false_positives"]


def test_phase14_training_function_emits_same_financial_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "ARTIFACTS_DIR", str(tmp_path))
    result = trainer.train_and_evaluate_model(n_samples=200, random_state=19)
    costs = result["cost_analysis"]
    assert "average_legitimate_transaction_value_usd" in costs
    assert "average_fraudulent_transaction_value_usd" in costs
    assert "false_positive_transaction_value_usd" in costs
    assert "false_negative_transaction_value_usd" in costs

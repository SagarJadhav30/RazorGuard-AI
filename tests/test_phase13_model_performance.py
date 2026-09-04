"""Contract tests for the Phase 13 model performance dashboard."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
COMPONENT = (ROOT / "frontend" / "src" / "components" / "ModelPerformancePage.jsx").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")


def test_phase13_model_page_is_wired_to_navigation():
    assert "ModelPerformancePage" in APP
    assert "metrics={data.metrics}" in APP


def test_phase13_displays_all_requested_held_out_metrics():
    for label in (
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "False Positive Rate",
        "False Negative Rate",
    ):
        assert label in COMPONENT
    assert "Held-out test" in COMPONENT


def test_phase13_displays_evaluation_artifacts_and_curves():
    for label in (
        "TRAINING",
        "VALIDATION",
        "HELD-OUT TEST",
        "Confusion matrix",
        "Precision / Recall tradeoff",
        "Threshold analysis",
        "Fraud class distribution",
    ):
        assert label in COMPONENT
    assert "metrics.confusion_matrix" in COMPONENT
    assert "metrics.pr_curve_data" in COMPONENT
    assert "metrics.roc_curve_data" in COMPONENT


def test_phase13_derives_rates_and_distribution_from_backend_data():
    assert "true_negatives" in COMPONENT
    assert "false_positives" in COMPONENT
    assert "false_negatives" in COMPONENT
    assert "true_positives" in COMPONENT
    assert "fp / (tn + fp)" in COMPONENT
    assert "fn / (fn + tp)" in COMPONENT
    assert "metrics.test_fraud_count" in COMPONENT
    assert "metrics.test_legit_count" in COMPONENT


def test_phase13_metadata_never_falls_back_to_fake_values():
    for label in ("Model name", "Version", "Training date", "Dataset version", "Feature count", "Test set size"):
        assert label in COMPONENT
    assert "metrics.model_version" in COMPONENT
    assert "metrics.training_date" in COMPONENT
    assert "metrics.dataset_version" in COMPONENT
    assert "Unavailable" in COMPONENT


def test_phase13_has_loading_and_empty_states():
    assert "Loading held-out model evaluation" in COMPONENT
    assert "No model evaluation data available" in COMPONENT
    assert "No precision-recall curve data" in COMPONENT
    assert "No threshold curve data" in COMPONENT

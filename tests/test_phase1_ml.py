"""
RazorGuard AI - Phase 1 ML Pipeline & Held-Out Test Set Metrics Unit Tests
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from backend.ml.data_generator import (
    generate_payment_dataset, get_preset_scenarios, FEATURE_NAMES
)
from backend.ml.trainer import (
    train_and_evaluate_model, load_trained_model, load_metrics_summary, ARTIFACTS_DIR
)


def test_data_generator_schema_and_distribution():
    """Verify synthetic dataset structure, feature names, and non-empty rows."""
    df = generate_payment_dataset(n_samples=1000, fraud_rate=0.10, random_state=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert "is_fraud" in df.columns
    for feat in FEATURE_NAMES:
        assert feat in df.columns, f"Missing feature: {feat}"

    # Check target class distribution
    fraud_count = df["is_fraud"].sum()
    assert fraud_count == 100, f"Expected 100 fraud rows, got {fraud_count}"

    # Feature range checks
    assert (df["amount"] > 0).all()
    assert (df["device_trust_score"] >= 0.0).all() and (df["device_trust_score"] <= 1.0).all()
    assert (df["hour_of_day"] >= 0).all() and (df["hour_of_day"] <= 23).all()


def test_preset_scenarios():
    """Verify preset simulation scenarios contain all required features."""
    presets = get_preset_scenarios()
    assert len(presets) >= 4

    for key, scenario in presets.items():
        assert "name" in scenario
        assert "description" in scenario
        assert "data" in scenario
        data = scenario["data"]
        for feat in FEATURE_NAMES:
            assert feat in data, f"Preset {key} missing feature {feat}"


def test_model_training_and_heldout_metrics():
    """Verify model trains cleanly and calculates non-fake held-out test set metrics."""
    res = train_and_evaluate_model(n_samples=2000, fraud_rate=0.08, random_state=42)

    assert "metrics" in res
    metrics = res["metrics"]

    # Verify realistic non-trivial metrics on held-out set
    assert 0.50 <= metrics["roc_auc"] <= 1.0
    assert 0.10 <= metrics["precision"] <= 1.0
    assert 0.10 <= metrics["recall"] <= 1.0
    assert 0.10 <= metrics["f1_score"] <= 1.0

    # Verify confusion matrix math
    cm = res["confusion_matrix"]
    total_test = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
    assert total_test == res["n_heldout_test_samples"]

    # Verify cost matrix math
    cost = res["cost_analysis"]
    assert cost["baseline_unmitigated_fraud_cost_usd"] > 0
    assert cost["net_financial_savings_usd"] > 0

    # Check serialized artifact files exist
    assert os.path.exists(os.path.join(ARTIFACTS_DIR, "model.joblib"))
    assert os.path.exists(os.path.join(ARTIFACTS_DIR, "metrics.json"))
    assert os.path.exists(os.path.join(ARTIFACTS_DIR, "feature_metadata.json"))


def test_load_metrics_summary():
    """Verify saved metrics artifact loading."""
    metrics = load_metrics_summary()
    assert "metrics" in metrics
    assert "roc_auc" in metrics["metrics"]
    assert "roc_curve_data" in metrics
    assert len(metrics["roc_curve_data"]) > 0

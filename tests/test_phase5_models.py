"""
RazorGuard AI - Phase 5 Model Training & Predictor Unit Tests
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from ml.utils.data_generator import generate_synthetic_fraud_dataset
from ml.training.trainer import train_and_compare_models, calculate_evaluation_metrics
from ml.inference.predictor import RazorGuardPredictor


def test_metric_calculation():
    """Verify evaluation metric computation math."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = calculate_evaluation_metrics(y_true, y_prob, threshold=0.50)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0


def test_model_training_and_artifact_generation():
    """Verify multi-model training pipeline runs cleanly and creates all required artifacts."""
    results = train_and_compare_models(data_path="data/processed/processed_fraud_transactions.csv")

    assert "winning_model_name" in results
    assert len(results["evaluations"]) >= 3

    # Check serialized artifacts exist
    assert os.path.exists(os.path.join("models", "classifier.joblib"))
    assert os.path.exists(os.path.join("models", "isolation_forest.joblib"))
    assert os.path.exists(os.path.join("models", "model_metadata.json"))
    assert os.path.exists(os.path.join("docs", "model_comparison.md"))


def test_predictor_inference():
    """Verify RazorGuardPredictor loads artifacts and outputs valid probabilities & anomaly scores."""
    df_sample = generate_synthetic_fraud_dataset(n_samples=20, fraud_rate=0.10, random_seed=42)

    predictor = RazorGuardPredictor(models_dir="models").load()

    probs = predictor.predict_fraud_probability(df_sample)
    assert len(probs) == 20
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    anom_scores = predictor.predict_anomaly_score(df_sample)
    assert len(anom_scores) == 20
    assert (anom_scores >= 0.0).all() and (anom_scores <= 100.0).all()

    single_tx = df_sample.iloc[0].to_dict()
    res = predictor.assess_raw_transaction(single_tx)

    assert "fraud_probability" in res
    assert "anomaly_score" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert 0.0 <= res["anomaly_score"] <= 100.0

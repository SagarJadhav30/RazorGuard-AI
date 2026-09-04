"""
RazorGuard AI - Phase 2 Dataset Generation & Validation Unit Tests
"""

import os
import json
import pytest
import pandas as pd

from ml.utils.data_generator import generate_synthetic_fraud_dataset, FEATURE_COLUMNS
from ml.preprocessing.validator import validate_payment_dataset


def test_dataset_generation_shape_and_columns():
    """Verify dataset generates correct row count, 27 columns, and exact schema."""
    df = generate_synthetic_fraud_dataset(n_samples=5000, fraud_rate=0.04, random_seed=42)

    assert df.shape == (5000, 27)
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_reproducibility():
    """Verify random seed produces identical dataset outputs."""
    df1 = generate_synthetic_fraud_dataset(n_samples=1000, fraud_rate=0.04, random_seed=42)
    df2 = generate_synthetic_fraud_dataset(n_samples=1000, fraud_rate=0.04, random_seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_class_imbalance():
    """Verify realistic target class imbalance (approx 3% to 6% fraud)."""
    df = generate_synthetic_fraud_dataset(n_samples=10000, fraud_rate=0.04, random_seed=42)
    fraud_rate = df["is_fraud"].mean()

    assert 0.03 <= fraud_rate <= 0.055, f"Unexpected fraud rate: {fraud_rate}"


def test_dataset_quality_audit():
    """Verify missing values, duplicate IDs, invalid ranges, and leakage audit."""
    df = generate_synthetic_fraud_dataset(n_samples=5000, fraud_rate=0.04, random_seed=42)
    report = validate_payment_dataset(df)

    assert report["validation_status"] == "PASSED"
    assert report["data_integrity"]["total_missing_values"] == 0
    assert report["data_integrity"]["duplicate_transaction_ids"] == 0
    assert report["data_integrity"]["total_invalid_values"] == 0
    assert len(report["target_leakage_audit"]["high_leakage_features_detected"]) == 0


def test_data_dictionary_file_exists():
    """Verify data dictionary documentation exists."""
    dict_path = os.path.join("docs", "data_dictionary.md")
    assert os.path.exists(dict_path), "docs/data_dictionary.md must exist"

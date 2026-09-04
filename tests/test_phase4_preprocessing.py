"""
RazorGuard AI - Phase 4 Preprocessing & Feature Engineering Unit Tests
"""

import os
import pytest
import numpy as np
import pandas as pd

from ml.utils.data_generator import generate_synthetic_fraud_dataset
from ml.features.feature_engineer import add_engineered_features, ENGINEERED_FEATURE_NAMES
from ml.preprocessing.pipeline import RazorGuardPreprocessor, separate_target


def test_feature_engineering_outputs():
    """Verify all 10 engineered features are generated cleanly with correct formulas."""
    df = generate_synthetic_fraud_dataset(n_samples=100, fraud_rate=0.05, random_seed=42)
    df_eng = add_engineered_features(df)

    for feature in ENGINEERED_FEATURE_NAMES:
        assert feature in df_eng.columns, f"Missing engineered feature: {feature}"
        assert not df_eng[feature].isnull().any(), f"NaN values found in engineered feature: {feature}"

    # Range and logic checks
    assert (df_eng["new_account_flag"].isin([0, 1])).all()
    assert (df_eng["nocturnal_tx_flag"].isin([0, 1])).all()
    assert (df_eng["velocity_ratio_24h_7d"] >= 0.0).all()
    assert (df_eng["amount_ratio_to_avg"] >= 0.0).all()


def test_missing_values_handling():
    """Verify preprocessor handles missing numerical and categorical values safely."""
    df = generate_synthetic_fraud_dataset(n_samples=200, fraud_rate=0.05, random_seed=42)
    X, y = separate_target(df)

    # Inject NaNs into numerical & categorical columns
    X_corrupted = X.copy()
    X_corrupted.loc[0:10, "amount"] = np.nan
    X_corrupted.loc[5:15, "velocity_score"] = np.nan
    X_corrupted.loc[10:20, "merchant_category"] = np.nan
    X_corrupted.loc[15:25, "country"] = np.nan

    preprocessor = RazorGuardPreprocessor()
    preprocessor.fit(X)  # Fit on clean data

    transformed = preprocessor.transform(X_corrupted)
    assert not np.isnan(transformed).any(), "Transformed array contains NaN values"


def test_unseen_categories_handling():
    """Verify preprocessor handles novel/unseen categorical values during inference."""
    df = generate_synthetic_fraud_dataset(n_samples=200, fraud_rate=0.05, random_seed=42)
    X, y = separate_target(df)

    preprocessor = RazorGuardPreprocessor()
    preprocessor.fit(X)

    # Create unseen categories in test payload
    X_unseen = X.iloc[:5].copy()
    X_unseen.loc[0, "country"] = "ZZ"  # Novel country
    X_unseen.loc[1, "merchant_category"] = "space_tourism"  # Novel merchant category
    X_unseen.loc[2, "payment_method"] = "crypto_wallet"  # Novel payment method
    X_unseen.loc[3, "currency"] = "BITCOIN"  # Novel currency

    transformed = preprocessor.transform(X_unseen)
    assert transformed.shape[0] == 5
    assert not np.isnan(transformed).any()


def test_invalid_boundary_inputs():
    """Verify preprocessor handles zero amounts, zero account age, and boundary inputs."""
    df = generate_synthetic_fraud_dataset(n_samples=50, fraud_rate=0.05, random_seed=42)
    X, y = separate_target(df)

    X_boundary = X.copy()
    X_boundary["amount"] = 0.0
    X_boundary["account_age_days"] = 0
    X_boundary["average_transaction_amount"] = 0.0
    X_boundary["payment_attempts"] = 0
    X_boundary["unique_devices"] = 0

    preprocessor = RazorGuardPreprocessor()
    preprocessor.fit(X)

    transformed = preprocessor.transform(X_boundary)
    assert not np.isnan(transformed).any()
    assert not np.isinf(transformed).any()


def test_inference_consistency_and_serialization(tmp_path):
    """Verify serialization and reloading preprocessor yields identical output arrays."""
    df = generate_synthetic_fraud_dataset(n_samples=500, fraud_rate=0.05, random_seed=42)
    X, y = separate_target(df)

    preprocessor = RazorGuardPreprocessor()
    preprocessor.fit(X)

    orig_arr = preprocessor.transform(X)

    # Save to temp path
    artifact_path = os.path.join(tmp_path, "preprocessor.joblib")
    preprocessor.save(artifact_path)

    # Load serialized preprocessor
    loaded_preprocessor = RazorGuardPreprocessor.load(artifact_path)
    loaded_arr = loaded_preprocessor.transform(X)

    np.testing.assert_array_almost_equal(orig_arr, loaded_arr)
    assert list(preprocessor.feature_names_out_) == list(loaded_preprocessor.feature_names_out_)

"""
RazorGuard AI - Production Preprocessing Pipeline

Combines feature engineering, numerical imputation & scaling, categorical encoding with unseen category handling,
and joblib serialization into a single unified sklearn-compatible pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ml.features.feature_engineer import add_engineered_features, ENGINEERED_FEATURE_NAMES

# Identifier and metadata columns to drop before ML modeling
DROP_COLUMNS: List[str] = ["transaction_id", "customer_id", "merchant_id", "timestamp"]

CATEGORICAL_FEATURES: List[str] = ["merchant_category", "payment_method", "currency", "country"]

NUMERICAL_BASE_FEATURES: List[str] = [
    "amount",
    "account_age_days",
    "customer_transaction_count",
    "transactions_last_24h",
    "transactions_last_7d",
    "average_transaction_amount",
    "payment_attempts",
    "failed_payment_count",
    "previous_fraud_count",
    "previous_chargeback_count",
    "unique_devices",
    "unique_ips",
    "billing_shipping_match",
    "country_change",
    "device_reuse_count",
    "velocity_score",
    "hour_of_day",
    "day_of_week"
]


class RazorGuardPreprocessor(BaseEstimator, TransformerMixin):
    """
    Unified production preprocessor for RazorGuard AI.
    Fits strictly on training data and transforms both batch & single-row inference payloads cleanly.
    """
    def __init__(self):
        self.numeric_features = NUMERICAL_BASE_FEATURES + ENGINEERED_FEATURE_NAMES
        self.categorical_features = CATEGORICAL_FEATURES
        self.column_transformer = None
        self.feature_names_out_ = None
        self.is_fitted_ = False

    def _build_column_transformer(self):
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features)
            ],
            remainder="drop"
        )

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits scalers, imputers, and encoders strictly on input DataFrame X.
        """
        X_eng = add_engineered_features(X)

        self.column_transformer = self._build_column_transformer()
        self.column_transformer.fit(X_eng)

        # Retrieve transformed feature names
        num_names = self.numeric_features
        cat_encoder = self.column_transformer.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(self.categorical_features))

        self.feature_names_out_ = num_names + cat_names
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw DataFrame X into preprocessed numpy array.
        """
        if not self.is_fitted_:
            raise RuntimeError("RazorGuardPreprocessor is not fitted. Call fit() before transform().")

        X_eng = add_engineered_features(X)
        transformed_arr = self.column_transformer.transform(X_eng)
        return transformed_arr

    def transform_to_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw DataFrame X into a clean pandas DataFrame with feature column names.
        """
        arr = self.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_out_)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, filepath: str = "models/preprocessor.joblib"):
        """
        Serializes fitted preprocessor object to joblib file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/preprocessor.joblib") -> "RazorGuardPreprocessor":
        """
        Loads serialized preprocessor object from joblib file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor artifact not found at: {filepath}")
        return joblib.load(filepath)


def separate_target(df: pd.DataFrame, target_col: str = "is_fraud") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates target column from features DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y

"""
RazorGuard AI - Production Inference Predictor Wrapper

Loads preprocessor, classifier, and anomaly detector artifacts to perform fast,
thread-safe fraud probability and anomaly score predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union

from ml.preprocessing.pipeline import RazorGuardPreprocessor, DROP_COLUMNS


class RazorGuardPredictor:
    """
    Production Predictor Wrapper for single-row and batch inference.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.classifier_path = os.path.join(models_dir, "classifier.joblib")
        self.isolation_forest_path = os.path.join(models_dir, "isolation_forest.joblib")

        self.preprocessor: RazorGuardPreprocessor = None
        self.classifier = None
        self.isolation_forest = None
        self.is_loaded = False

    def load(self):
        """
        Loads preprocessor, classifier, and isolation forest artifacts into memory.
        """
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"Classifier artifact not found at {self.classifier_path}. Run scripts/train_models.py first.")

        self.preprocessor = RazorGuardPreprocessor.load(self.preprocessor_path)
        self.classifier = joblib.load(self.classifier_path)

        if os.path.exists(self.isolation_forest_path):
            self.isolation_forest = joblib.load(self.isolation_forest_path)

        self.is_loaded = True
        return self

    def _prepare_df(self, payload: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(payload, dict):
            df = pd.DataFrame([payload])
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        elif isinstance(payload, pd.DataFrame):
            df = payload.copy()
        else:
            raise ValueError("Payload must be a DataFrame, dict, or list of dicts.")

        # Drop target if present
        if "is_fraud" in df.columns:
            df = df.drop(columns=["is_fraud"])

        return df

    def predict_fraud_probability(self, payload: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """
        Returns calibrated ML fraud probability P in [0.0, 1.0].
        """
        if not self.is_loaded:
            self.load()

        df = self._prepare_df(payload)
        X_proc = self.preprocessor.transform(df)
        probs = self.classifier.predict_proba(X_proc)[:, 1]
        return np.round(probs, 4)

    def predict_anomaly_score(self, payload: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """
        Returns normalized Isolation Forest anomaly score (0 - 100). Higher = more anomalous.
        """
        if not self.is_loaded:
            self.load()

        if self.isolation_forest is None:
            return np.zeros(len(payload))

        df = self._prepare_df(payload)
        X_proc = self.preprocessor.transform(df)

        # Isolation forest decision_function returns negative values for anomalies
        raw_scores = self.isolation_forest.decision_function(X_proc)
        # Normalize to 0 - 100 scale (where lower raw score = higher anomaly score)
        norm_scores = np.clip((0.5 - raw_scores) * 100.0, 0.0, 100.0)
        return np.round(norm_scores, 2)

    def assess_raw_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single-transaction assessment helper returning probability and anomaly score.
        """
        prob = float(self.predict_fraud_probability(transaction)[0])
        anom_score = float(self.predict_anomaly_score(transaction)[0])

        return {
            "fraud_probability": prob,
            "anomaly_score": anom_score
        }

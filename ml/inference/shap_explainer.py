"""
RazorGuard AI - SHAP Feature Attribution Explainer

Calculates exact local SHAP feature attributions for any payment transaction.
Translates mathematical SHAP values into positive risk factors and protective mitigating factors.
"""

import os
import joblib
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from ml.preprocessing.pipeline import RazorGuardPreprocessor

FEATURE_HUMAN_LABELS: Dict[str, str] = {
    "amount": "Transaction Amount ($)",
    "account_age_days": "Account Tenure (Days)",
    "customer_transaction_count": "Historical Transaction Count",
    "transactions_last_24h": "24-Hour Transaction Frequency",
    "transactions_last_7d": "7-Day Transaction Frequency",
    "average_transaction_amount": "Historical Average Transaction Amount",
    "payment_attempts": "Checkout Payment Attempts",
    "failed_payment_count": "Failed Payment Attempt Count",
    "previous_fraud_count": "Previous Confirmed Fraud Incidents",
    "previous_chargeback_count": "Previous Historical Chargebacks",
    "unique_devices": "Unique Devices (30 Days)",
    "unique_ips": "Unique IP Addresses (30 Days)",
    "billing_shipping_match": "Billing/Shipping Address Match",
    "country_change": "IP Billing Country Mismatch",
    "device_reuse_count": "Device Reuse Count across Accounts",
    "velocity_score": "Real-Time Velocity Risk Score",
    "hour_of_day": "Transaction Local Hour of Day",
    "day_of_week": "Transaction Day of Week",
    "velocity_ratio_24h_7d": "24h/7d Transaction Velocity Spike Ratio",
    "amount_ratio_to_avg": "Transaction Amount to Average Ratio",
    "payment_failure_ratio": "Payment Failure Ratio",
    "device_sharing_risk": "Device Sharing Risk Index",
    "ip_device_ratio": "IP to Device Hopping Ratio",
    "new_account_flag": "New Account Indicator (< 14 Days)",
    "account_age_log": "Log Account Age",
    "nocturnal_tx_flag": "Nocturnal Transaction Hours (1 AM - 4 AM)",
    "location_mismatch_risk": "Location & Address Mismatch Composite Risk",
    "history_risk_score": "Historical Pre-Event Fraud Risk Score"
}


class RazorGuardSHAPExplainer:
    """
    Computes exact SHAP attributions using background dataset from preprocessor & classifier.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.classifier_path = os.path.join(models_dir, "classifier.joblib")

        self.preprocessor: RazorGuardPreprocessor = None
        self.classifier = None
        self.explainer = None
        self.feature_names: List[str] = []
        self.is_loaded = False

    def load(self):
        """
        Loads preprocessor and classifier, initializes SHAP explainer.
        """
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"Classifier not found at {self.classifier_path}")

        self.preprocessor = RazorGuardPreprocessor.load(self.preprocessor_path)
        self.classifier = joblib.load(self.classifier_path)
        self.feature_names = list(self.preprocessor.feature_names_out_)

        # Create dummy background matrix for SHAP explainer
        sample_df = pd.DataFrame([{
            "amount": 50.0, "currency": "USD", "merchant_category": "retail", "payment_method": "credit_card",
            "country": "US", "account_age_days": 100, "customer_transaction_count": 10, "transactions_last_24h": 1,
            "transactions_last_7d": 5, "average_transaction_amount": 50.0, "payment_attempts": 1,
            "failed_payment_count": 0, "previous_fraud_count": 0, "previous_chargeback_count": 0,
            "unique_devices": 1, "unique_ips": 1, "billing_shipping_match": 1, "country_change": 0,
            "device_reuse_count": 5, "velocity_score": 10.0, "hour_of_day": 12, "day_of_week": 2
        }])
        bg_proc = self.preprocessor.transform(sample_df)

        try:
            # Use SHAP LinearExplainer for LogisticRegression or TreeExplainer for Tree models
            self.explainer = shap.Explainer(self.classifier, bg_proc)
        except Exception:
            self.explainer = shap.KernelExplainer(self.classifier.predict_proba, bg_proc)

        self.is_loaded = True
        return self

    def compute_shap_attributions(self, df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes SHAP feature attribution values for input raw DataFrame.
        Returns (shap_values, preprocessed_feature_values).
        """
        if not self.is_loaded:
            self.load()

        X_proc = self.preprocessor.transform(df_raw)
        shap_res = self.explainer(X_proc)

        if hasattr(shap_res, "values"):
            vals = shap_res.values
            # If 3D array (for multi-class or binary probabilities), extract positive class (idx 1)
            if vals.ndim == 3:
                vals = vals[:, :, 1]
        else:
            vals = np.array(shap_res)

        return vals, X_proc

    def explain_single_transaction(
        self,
        df_raw: pd.DataFrame,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Extracts top K risk-increasing and risk-decreasing feature attributions for single transaction.
        """
        shap_vals, X_proc = self.compute_shap_attributions(df_raw)
        shap_row = shap_vals[0]
        proc_row = X_proc[0]

        factors = []
        for i, feat_name in enumerate(self.feature_names):
            s_val = float(shap_row[i])
            if abs(s_val) < 1e-5:
                continue

            # Human friendly name lookup
            clean_feat = feat_name.split("__")[-1] if "__" in feat_name else feat_name
            human_label = FEATURE_HUMAN_LABELS.get(clean_feat, clean_feat.replace("_", " ").title())

            direction = "risk_increasing" if s_val > 0 else "risk_decreasing"

            factors.append({
                "feature": clean_feat,
                "feature_name_human": human_label,
                "feature_value": round(float(proc_row[i]), 4),
                "shap_value": round(s_val, 4),
                "impact_direction": direction,
                "abs_impact": abs(s_val)
            })

        # Sort factors by absolute impact descending
        factors_sorted = sorted(factors, key=lambda x: x["abs_impact"], reverse=True)

        risk_factors = [f for f in factors_sorted if f["impact_direction"] == "risk_increasing"][:top_k]
        protective_factors = [f for f in factors_sorted if f["impact_direction"] == "risk_decreasing"][:top_k]

        return {
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "all_attributions": {f["feature"]: f["shap_value"] for f in factors}
        }

"""
RazorGuard AI - Feature Engineering Transformer

Constructs domain-informed behavioral risk features from raw payment transaction attributes.
Guarantees zero target leakage and supports both single-row inference and batch datasets.
"""

import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin 
from typing import List, Dict, Any


ENGINEERED_FEATURE_NAMES: List[str] = [ 
    "velocity_ratio_24h_7d",
    "amount_ratio_to_avg",
    "payment_failure_ratio",
    "device_sharing_risk",
    "ip_device_ratio",
    "new_account_flag",
    "account_age_log",
    "nocturnal_tx_flag", 
    "location_mismatch_risk",
    "history_risk_score"
]

ENGINEERED_FEATURE_DOCS: Dict[str, str] = {
    "velocity_ratio_24h_7d": "Ratio of 24h transaction volume vs daily average of trailing 7d volume. Detects velocity spikes.",
    "amount_ratio_to_avg": "Ratio of current transaction amount vs customer's historical average amount. Detects amount anomalies.",
    "payment_failure_ratio": "Ratio of failed payment attempts to total attempts for checkout. Detects brute-force authorization retries.",
    "device_sharing_risk": "Ratio of device reuse count across accounts to customer's unique device count. Detects botnet device sharing.",
    "ip_device_ratio": "Ratio of unique IP addresses used to unique devices. Detects proxy/VPN IP hopping.",
    "new_account_flag": "Binary indicator (1 if account age <= 14 days else 0). Identifies high-risk newly registered accounts.",
    "account_age_log": "Log1p transformation of account_age_days. Normalizes heavily skewed tenure distribution.",
    "nocturnal_tx_flag": "Binary indicator (1 if transaction hour in 1 AM - 4 AM else 0). Identifies off-hours automated attacks.",
    "location_mismatch_risk": "Composite risk score combining country change and billing/shipping address mismatch.",
    "history_risk_score": "Weighted historical risk score combining previous fraud and chargeback counts."
}


def add_engineered_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Pure transformation function that creates 10 domain engineered features.
    """
    df = df_input.copy()

    # 1. Transaction Velocity Ratio (24h vs 7d daily average)
    daily_avg_7d = (df["transactions_last_7d"] / 7.0) + 0.1
    df["velocity_ratio_24h_7d"] = np.round(df["transactions_last_24h"] / daily_avg_7d, 4)

    # 2. Amount Deviation from Customer Average
    df["amount_ratio_to_avg"] = np.round(df["amount"] / (df["average_transaction_amount"] + 1.0), 4)

    # 3. Payment Failure Ratio
    attempts = np.maximum(df["payment_attempts"], 1.0)
    df["payment_failure_ratio"] = np.round(df["failed_payment_count"] / attempts, 4)

    # 4. Device Sharing Risk
    devices = np.maximum(df["unique_devices"], 1.0) 
    df["device_sharing_risk"] = np.round(df["device_reuse_count"] / devices, 4)

    # 5. IP Reuse Ratio
    df["ip_device_ratio"] = np.round(df["unique_ips"] / devices, 4)

    # 6. Account Age Risk (Flag & Log)
    df["new_account_flag"] = (df["account_age_days"] <= 14).astype(int)
    df["account_age_log"] = np.round(np.log1p(np.maximum(df["account_age_days"], 0)), 4)

    # 7. Unusual Transaction Hour (Nocturnal flag: 1 AM - 4 AM)
    df["nocturnal_tx_flag"] = df["hour_of_day"].isin([1, 2, 3, 4]).astype(int)

    # 8. Billing/Shipping Mismatch & Location Risk
    # country_change = 1 (IP mismatch), billing_shipping_match = 0 (address mismatch)
    mismatch_flag = (df["billing_shipping_match"] == 0).astype(int)
    df["location_mismatch_risk"] = df["country_change"] * 1.5 + mismatch_flag * 1.0

    # 9. Previous Fraud History Composite Score
    df["history_risk_score"] = np.round(
        df["previous_fraud_count"] * 2.0 + df["previous_chargeback_count"] * 1.5, 4
    )

    return df


class PaymentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer wrapper for feature engineering.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_engineered_features(X)

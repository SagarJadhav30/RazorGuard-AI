"""
RazorGuard AI - Dataset Validation & Quality Report Generator

Performs rigorous initial validation: missing values, duplicate IDs, invalid ranges,
class distribution, feature correlations, target leakage checks, and generates docs/dataset_report.json.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any


def validate_payment_dataset(df: pd.DataFrame, report_output_dir: str = "docs") -> Dict[str, Any]:
    """
    Executes full quality assurance audit on the payment fraud dataset.
    """
    os.makedirs(report_output_dir, exist_ok=True)

    n_rows, n_cols = df.shape
    missing_dict = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    # Duplicate IDs Check
    duplicate_tx_ids = int(df["transaction_id"].duplicated().sum())

    # Class Distribution
    class_counts = df["is_fraud"].value_counts().to_dict()
    legit_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_rate_pct = round((fraud_count / n_rows) * 100.0, 3)

    # Range & Invalid Values Verification
    invalid_checks = {
        "negative_amounts": int((df["amount"] <= 0).sum()),
        "invalid_hours": int(((df["hour_of_day"] < 0) | (df["hour_of_day"] > 23)).sum()),
        "invalid_days": int(((df["day_of_week"] < 0) | (df["day_of_week"] > 6)).sum()),
        "invalid_billing_match": int((~df["billing_shipping_match"].isin([0, 1])).sum()),
        "invalid_country_change": int((~df["country_change"].isin([0, 1])).sum()),
        "negative_account_age": int((df["account_age_days"] < 0).sum())
    }
    total_invalid = sum(invalid_checks.values())

    # Target Leakage Audit (Calculate Pearson correlation with target for numeric features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    leakage_correlations = {}
    high_leakage_features = []

    for col in numeric_cols:
        if col != "is_fraud":
            corr = float(df[col].corr(df["is_fraud"]))
            leakage_correlations[col] = round(corr, 4)
            if abs(corr) > 0.85:
                high_leakage_features.append(col)

    # Statistical Feature Distributions
    feature_stats = {}
    for col in numeric_cols:
        feature_stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "median": round(float(df[col].median()), 4),
            "max": round(float(df[col].max()), 4)
        }

    validation_status = "PASSED" if (total_missing == 0 and duplicate_tx_ids == 0 and total_invalid == 0 and len(high_leakage_features) == 0) else "FAILED"

    report = {
        "dataset_name": "RazorGuard Synthetic Payment Fraud Dataset",
        "validation_status": validation_status,
        "dataset_shape": {
            "num_records": n_rows,
            "num_features": n_cols
        },
        "class_distribution": {
            "legitimate_count": legit_count,
            "fraud_count": fraud_count,
            "fraud_rate_percentage": fraud_rate_pct
        },
        "data_integrity": {
            "total_missing_values": total_missing,
            "duplicate_transaction_ids": duplicate_tx_ids,
            "total_invalid_values": total_invalid,
            "invalid_range_breakdown": invalid_checks
        },
        "target_leakage_audit": {
            "high_leakage_features_detected": high_leakage_features,
            "max_feature_correlation": max(leakage_correlations.values(), key=abs) if leakage_correlations else 0.0,
            "feature_correlations_with_target": leakage_correlations
        },
        "feature_summary_statistics": feature_stats
    }

    # Save JSON Report
    report_path = os.path.join(report_output_dir, "dataset_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report

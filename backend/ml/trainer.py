"""
RazorGuard AI - ML Model Trainer & Held-Out Test Set Evaluator

Trains fraud detection classifier (LightGBM/RandomForest) strictly on 80% training data,
and evaluates performance metrics and financial impact strictly on a 20% held-out test set.
Serializes model artifacts and metrics.json for production API usage.
"""

import os
import json
import joblib
import numpy as np 
import pandas as pd
from typing import Dict, Any, Tuple
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from backend.ml.data_generator import generate_payment_dataset, FEATURE_NAMES, FEATURE_LABELS

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def train_and_evaluate_model(
    n_samples: int = 10000,
    fraud_rate: float = 0.08, 
    random_state: int = 42,
    cost_per_false_negative: float = 450.0,  # Avg dollar loss per missed fraud
    cost_per_false_positive: float = 15.0    # Friction/support cost per false block
) -> Dict[str, Any]:
    """
    Executes full ML training pipeline and evaluates strictly on held-out test set.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Generate Dataset
    df = generate_payment_dataset(n_samples=n_samples, fraud_rate=fraud_rate, random_state=random_state)

    X = df[FEATURE_NAMES]
    y = df["is_fraud"]

    # 2. Train/Test Split (80% Train, 20% Held-Out Test Set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # 3. Model Training
    if HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1
        )
        model_type = "LightGBM Classifier"
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1
        )
        model_type = "RandomForest Classifier"

    model.fit(X_train, y_train)

    # 4. Strictly Held-Out Test Set Evaluation
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds_default = (y_probs >= 0.50).astype(int)

    precision = float(precision_score(y_test, y_preds_default))
    recall = float(recall_score(y_test, y_preds_default))
    f1 = float(f1_score(y_test, y_preds_default))
    roc_auc = float(roc_auc_score(y_test, y_probs))
    pr_auc = float(average_precision_score(y_test, y_probs))

    cm = confusion_matrix(y_test, y_preds_default)
    tn, fp, fn, tp = [int(val) for val in cm.ravel()]

    # 5. ROC Curve Data Points
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_probs)
    # Downsample curve points for clean JSON export
    idx_sample = np.linspace(0, len(fpr) - 1, num=min(50, len(fpr)), dtype=int)
    roc_curve_points = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4), "threshold": round(float(roc_thresholds[i]), 4)}
        for i in idx_sample
    ]

    # 6. Precision-Recall Curve Data Points
    p_pts, r_pts, pr_thresholds = precision_recall_curve(y_test, y_probs)
    pr_idx_sample = np.linspace(0, len(p_pts) - 1, num=min(50, len(p_pts)), dtype=int)
    pr_curve_points = [
        {"precision": round(float(p_pts[i]), 4), "recall": round(float(r_pts[i]), 4)}
        for i in pr_idx_sample
    ]

    # 7. Financial Loss & Savings Matrix
    total_test_fraud_count = int(y_test.sum())
    total_test_legit_count = int((y_test == 0).sum())
    avg_legitimate_amount = float(X_test.loc[y_test == 0, "amount"].mean()) if total_test_legit_count else 0.0
    avg_fraud_amount = float(X_test.loc[y_test == 1, "amount"].mean()) if total_test_fraud_count else 0.0
    false_positive_amount = float(X_test.loc[(y_test == 0) & (y_preds_default == 1), "amount"].sum())
    false_negative_amount = float(X_test.loc[(y_test == 1) & (y_preds_default == 0), "amount"].sum())

    baseline_no_detection_cost = float(total_test_fraud_count * cost_per_false_negative)
    model_false_negative_cost = float(fn * cost_per_false_negative)
    model_false_positive_cost = float(fp * cost_per_false_positive)
    model_total_loss_cost = model_false_negative_cost + model_false_positive_cost

    net_savings = baseline_no_detection_cost - model_total_loss_cost
    savings_percentage = round((net_savings / baseline_no_detection_cost) * 100.0, 2) if baseline_no_detection_cost > 0 else 0.0

    # 8. Compile Comprehensive Metrics Dict
    metrics_summary = {
        "model_type": model_type,
        "n_total_samples": len(df),
        "n_train_samples": len(X_train),
        "n_heldout_test_samples": len(X_test),
        "test_fraud_count": total_test_fraud_count,
        "test_legit_count": total_test_legit_count,
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "pr_auc": round(pr_auc, 4),
            "accuracy": round(float((tp + tn) / len(y_test)), 4)
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "cost_analysis": {
            "cost_per_false_negative_usd": cost_per_false_negative,
            "cost_per_false_positive_usd": cost_per_false_positive,
            "baseline_unmitigated_fraud_cost_usd": round(baseline_no_detection_cost, 2),
            "model_false_negative_cost_usd": round(model_false_negative_cost, 2),
            "model_false_positive_cost_usd": round(model_false_positive_cost, 2),
            "model_total_loss_cost_usd": round(model_total_loss_cost, 2),
            "net_financial_savings_usd": round(net_savings, 2),
            "savings_percentage": savings_percentage,
            "average_legitimate_transaction_value_usd": round(avg_legitimate_amount, 2),
            "average_fraudulent_transaction_value_usd": round(avg_fraud_amount, 2),
            "false_positive_transaction_value_usd": round(false_positive_amount, 2),
            "false_negative_transaction_value_usd": round(false_negative_amount, 2),
            "transactions_reviewed_without_ai": len(X_test),
            "transactions_reviewed_with_ai": len(X_test),
            "fraudulent_transactions_caught": tp,
            "legitimate_transactions_incorrectly_blocked": fp
        },
        "feature_names": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "roc_curve_data": roc_curve_points,
        "pr_curve_data": pr_curve_points
    }

    # 9. Save Artifacts
    model_filepath = os.path.join(ARTIFACTS_DIR, "model.joblib")
    joblib.dump(model, model_filepath)

    metrics_filepath = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_filepath, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    meta_filepath = os.path.join(ARTIFACTS_DIR, "feature_metadata.json")
    with open(meta_filepath, "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "feature_labels": FEATURE_LABELS,
            "model_type": model_type
        }, f, indent=2)

    return metrics_summary


def load_trained_model():
    """
    Loads saved ML model artifact.
    """
    model_filepath = os.path.join(ARTIFACTS_DIR, "model.joblib")
    if not os.path.exists(model_filepath):
        train_and_evaluate_model()
    return joblib.load(model_filepath)


def load_metrics_summary() -> Dict[str, Any]:
    """
    Loads serialized metrics.json.
    """
    metrics_filepath = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(metrics_filepath):
        train_and_evaluate_model()
    with open(metrics_filepath, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    print("Training RazorGuard AI ML Model & Calculating Held-Out Test Set Metrics...")
    res = train_and_evaluate_model()
    print(f"Model Type: {res['model_type']}")
    print(f"Held-Out Test Set ROC-AUC: {res['metrics']['roc_auc']}")
    print(f"Held-Out Test Set Precision: {res['metrics']['precision']}")
    print(f"Held-Out Test Set Recall: {res['metrics']['recall']}")
    print(f"Held-Out Test Set F1: {res['metrics']['f1_score']}")
    print(f"Net Financial Savings: ${res['cost_analysis']['net_financial_savings_usd']:,.2f} ({res['cost_analysis']['savings_percentage']}%)")

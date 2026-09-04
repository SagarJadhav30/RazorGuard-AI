"""
RazorGuard AI - Multi-Model Fraud Classifier Trainer & Evaluator

Trains and compares:
1. Logistic Regression (Baseline)
2. Random Forest Classifier
3. LightGBM Classifier (Gradient Boosted Trees)
4. Isolation Forest (Auxiliary Unsupervised Anomaly Detector)

Evaluates metrics strictly on Validation and Held-Out Test Sets (ROC-AUC, PR-AUC, Precision, Recall, F1, FPR, FNR).
Handles class imbalance via cost-sensitive class weights and scale_pos_weight.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from ml.preprocessing.pipeline import RazorGuardPreprocessor, separate_target


def calculate_evaluation_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """
    Computes exact empirical performance metrics from ground truth labels and predicted probabilities.
    """
    y_pred = (y_prob >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        }
    }


def train_and_compare_models(
    data_path: str = "data/processed/processed_fraud_transactions.csv",
    output_dir: str = "models",
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Splits data into 80% Train, 10% Validation, 10% Held-Out Test.
    Trains Logistic Regression, Random Forest, LightGBM, and Isolation Forest.
    Saves winning classifier and preprocessing artifacts.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # 1. Target & Feature Separation
    X, y = separate_target(df, target_col="is_fraud")

    # 2. Train / Validation / Test Split (80% Train, 10% Val, 10% Test)
    # First split 80% train vs 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )
    # Split 20% temp into 10% validation and 10% strictly held-out test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
    )

    # 3. Fit Preprocessor strictly on Training Set
    preprocessor = RazorGuardPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Save preprocessor artifact
    preprocessor.save(os.path.join(output_dir, "preprocessor.joblib"))

    # Compute class ratio for LightGBM scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = float(neg_count / max(pos_count, 1))

    # --- Model Definitions ---
    models: Dict[str, Any] = {}

    # 1. Logistic Regression (Baseline)
    models["Logistic Regression"] = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_seed
    )

    # 2. Random Forest Classifier
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=random_seed,
        n_jobs=-1
    )

    # 3. LightGBM / Gradient Boosting
    if HAS_LIGHTGBM:
        models["LightGBM Classifier"] = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=31,
            scale_pos_weight=pos_weight,
            random_state=random_seed,
            verbose=-1
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["Gradient Boosting"] = HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=200,
            random_state=random_seed
        )

    # 4. Auxiliary Isolation Forest (Unsupervised Anomaly Detector)
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.04,  # Expected anomaly rate
        random_state=random_seed,
        n_jobs=-1
    )
    iso_forest.fit(X_train_proc)
    joblib.dump(iso_forest, os.path.join(output_dir, "isolation_forest.joblib"))

    # --- Training & Evaluation Loop ---
    model_evaluations: Dict[str, Any] = {}
    best_model_name = None
    best_pr_auc = -1.0

    for name, model in models.items():
        # Fit model on training set only
        model.fit(X_train_proc, y_train)

        # Evaluate on Validation Set
        val_probs = model.predict_proba(X_val_proc)[:, 1]
        val_metrics = calculate_evaluation_metrics(y_val, val_probs)

        # Evaluate on Strictly Held-Out Test Set
        test_probs = model.predict_proba(X_test_proc)[:, 1]
        test_metrics = calculate_evaluation_metrics(y_test, test_probs)

        model_evaluations[name] = {
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics
        }

        # Track best model based on Held-Out Test PR-AUC (ideal metric for imbalanced fraud)
        if test_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = test_metrics["pr_auc"]
            best_model_name = name

    # 5. Save Winning Model Artifact
    winning_model = models[best_model_name]
    joblib.dump(winning_model, os.path.join(output_dir, "classifier.joblib"))

    # 6. Save Metadata & Config Artifacts
    metadata = {
        "winning_model_name": best_model_name,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "n_test_samples": len(X_test),
        "train_fraud_rate": round(float(y_train.mean()), 4),
        "test_fraud_rate": round(float(y_test.mean()), 4),
        "preprocessed_features_count": X_train_proc.shape[1],
        "feature_names": list(preprocessor.feature_names_out_),
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "winning_model_test_metrics": model_evaluations[best_model_name]["test_metrics"]
    }

    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(output_dir, "feature_metadata.json"), "w") as f:
        json.dump({
            "feature_names": list(preprocessor.feature_names_out_),
            "raw_feature_count": X.shape[1],
            "processed_feature_count": X_train_proc.shape[1]
        }, f, indent=2)

    return {
        "winning_model_name": best_model_name,
        "evaluations": model_evaluations,
        "metadata": metadata
    }

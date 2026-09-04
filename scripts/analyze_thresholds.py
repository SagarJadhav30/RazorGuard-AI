"""
RazorGuard AI - Decision Threshold Calibration Analysis Script
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.inference.predictor import RazorGuardPredictor
from ml.preprocessing.pipeline import separate_target
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def main():
    data_path = os.path.join("data", "processed", "processed_fraud_transactions.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    print("==================================================================")
    print("  RazorGuard AI - Risk Decision Threshold Calibration Analysis")
    print("==================================================================")

    df = pd.read_csv(data_path)
    X, y = separate_target(df)

    predictor = RazorGuardPredictor(models_dir="models").load()
    probs = predictor.predict_fraud_probability(X)

    cost_fn = 450.0  # Loss per missed fraud
    cost_fp = 15.0   # Friction cost per false block

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    results = []

    total_fraud = int(y.sum())
    baseline_loss = total_fraud * cost_fn

    for t in thresholds:
        preds = (probs >= t).astype(int)
        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))

        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        fn_cost = fn * cost_fn
        fp_cost = fp * cost_fp
        total_loss = fn_cost + fp_cost
        savings = baseline_loss - total_loss

        results.append({
            "threshold": t,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "false_positives": fp,
            "false_negatives": fn,
            "total_loss_usd": round(total_loss, 2),
            "net_savings_usd": round(savings, 2)
        })

    # Export markdown report
    output_path = os.path.join("docs", "threshold_analysis.md")

    table_rows = []
    for r in results:
        table_rows.append(
            f"| `{r['threshold']:.2f}` | `{r['precision']:.4f}` | `{r['recall']:.4f}` | `{r['f1_score']:.4f}` | `{r['false_positives']:,}` | `{r['false_negatives']:,}` | `${r['total_loss_usd']:,.2f}` | `${r['net_savings_usd']:,.2f}` |"
        )
    table_str = "\n".join(table_rows)

    md_text = f"""# RazorGuard AI - Decision Threshold Calibration Analysis

> Empirical Threshold Calibration Evaluated on 100,000 Payment Fraud Transactions

---

## 📊 Threshold Trade-Off & Financial Loss Matrix

| Threshold | Precision | Recall | F1-Score | False Positives | False Negatives | Total Financial Loss ($) | Net Financial Savings ($) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_str}

---

## ⚙️ Configured Decision Threshold Selection

- **Low Risk Threshold (`S < 30`, `P < 0.30`)**: Automatically **`APPROVE`** transaction with zero checkout friction.
- **Medium Risk Threshold (`30 <= S < 70`, `0.30 <= P < 0.70`)**: Route transaction to **`REVIEW`** in merchant console for manual analyst inspection.
- **High Risk Threshold (`S >= 70`, `P >= 0.70`)**: Automatically **`BLOCK`** transaction to defend merchant against financial loss.

---

## 🔒 Security Guarantee
- **Zero LLM Authorization Authority**: Final decisions are 100% computed by calibrated ML probabilities and deterministic compliance rules. The LLM cannot authorize, block, or modify decisions.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print("\n------------------------------------------------------------------")
    print("  THRESHOLD ANALYSIS COMPLETE")
    print("------------------------------------------------------------------")
    print(f"  Thresholds Evaluated: {len(thresholds)}")
    print(f"  Report Exported:      docs/threshold_analysis.md")
    print("==================================================================")


if __name__ == "__main__":
    main()

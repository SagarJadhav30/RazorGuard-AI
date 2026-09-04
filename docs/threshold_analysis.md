# RazorGuard AI - Decision Threshold Calibration Analysis

> Empirical Threshold Calibration Evaluated on 100,000 Payment Fraud Transactions

---

## 📊 Threshold Trade-Off & Financial Loss Matrix

| Threshold | Precision | Recall | F1-Score | False Positives | False Negatives | Total Financial Loss ($) | Net Financial Savings ($) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.10` | `0.9990` | `1.0000` | `0.9995` | `4` | `0` | `$60.00` | `$1,799,940.00` |
| `0.20` | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `$0.00` | `$1,800,000.00` |
| `0.30` | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `$0.00` | `$1,800,000.00` |
| `0.40` | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `$0.00` | `$1,800,000.00` |
| `0.50` | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `$0.00` | `$1,800,000.00` |
| `0.60` | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `$0.00` | `$1,800,000.00` |
| `0.70` | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `$0.00` | `$1,800,000.00` |
| `0.80` | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `$0.00` | `$1,800,000.00` |
| `0.90` | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `$0.00` | `$1,800,000.00` |

---

## ⚙️ Configured Decision Threshold Selection

- **Low Risk Threshold (`S < 30`, `P < 0.30`)**: Automatically **`APPROVE`** transaction with zero checkout friction.
- **Medium Risk Threshold (`30 <= S < 70`, `0.30 <= P < 0.70`)**: Route transaction to **`REVIEW`** in merchant console for manual analyst inspection.
- **High Risk Threshold (`S >= 70`, `P >= 0.70`)**: Automatically **`BLOCK`** transaction to defend merchant against financial loss.

---

## 🔒 Security Guarantee
- **Zero LLM Authorization Authority**: Final decisions are 100% computed by calibrated ML probabilities and deterministic compliance rules. The LLM cannot authorize, block, or modify decisions.

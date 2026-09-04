# RazorGuard AI - Engineered Features Documentation

This document specifies all **10 domain engineered features** created by `ml/features/feature_engineer.py` and processed by `ml/preprocessing/pipeline.py`.

---

## 🛠️ Engineered Feature Matrix

| Feature Name | Formula / Logic | Target Risk Rationale | Range / Output |
| :--- | :--- | :--- | :--- |
| `velocity_ratio_24h_7d` | `transactions_last_24h / (transactions_last_7d / 7.0 + 0.1)` | Identifies sudden transaction spikes relative to the customer's 7-day trailing baseline. | Float (`0.0` - `50.0+`) |
| `amount_ratio_to_avg` | `amount / (average_transaction_amount + 1.0)` | Measures transaction amount deviation relative to historical spending habits. | Float (`0.0` - `100.0+`) |
| `payment_failure_ratio` | `failed_payment_count / max(payment_attempts, 1)` | Detects authorization brute-force and card testing behavior. | Float (`0.0` - `1.0`) |
| `device_sharing_risk` | `device_reuse_count / max(unique_devices, 1)` | Identifies device sharing across multiple accounts (botnet/farm signal). | Float (`0.0` - `100.0+`) |
| `ip_device_ratio` | `unique_ips / max(unique_devices, 1)` | Detects proxy or VPN IP-hopping behavior relative to physical devices. | Float (`0.1` - `15.0+`) |
| `new_account_flag` | `1 if account_age_days <= 14 else 0` | Flags newly created accounts (< 14 days old) which suffer 65% of fraud attacks. | Binary (`0` or `1`) |
| `account_age_log` | `log1p(max(account_age_days, 0))` | Normalizes log-normal account tenure distribution for linear & tree models. | Float (`0.0` - `7.5`) |
| `nocturnal_tx_flag` | `1 if hour_of_day in [1, 2, 3, 4] else 0` | Captures off-hours automated overnight fraud scripts. | Binary (`0` or `1`) |
| `location_mismatch_risk` | `country_change * 1.5 + (1 - billing_shipping_match) * 1.0` | Combines billing/shipping address mismatch and IP country mismatch. | Float (`0.0` - `2.5`) |
| `history_risk_score` | `previous_fraud_count * 2.0 + previous_chargeback_count * 1.5` | Composite historical risk score from pre-event fraud and chargebacks. | Float (`0.0` - `17.5`) |

---

## 🔒 Preprocessing & Safety Guarantees
1. **Train-Only Fitting**: Scalers (`StandardScaler`), median imputers (`SimpleImputer`), and one-hot encoders (`OneHotEncoder`) are fit strictly on training splits.
2. **Unseen Category Safety**: `OneHotEncoder(handle_unknown='ignore')` maps novel country codes or merchant categories encountered during inference to zeros without throwing exceptions.
3. **Missing Value Imputation**: Missing numeric values are imputed with training medians; missing categorical values are filled with `'missing'`.
4. **Inference Consistency**: Input data passes through the exact same `PaymentFeatureEngineer` and `ColumnTransformer` during real-time REST API evaluation.

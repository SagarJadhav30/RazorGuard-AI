# RazorGuard AI - Payment Fraud Data Dictionary

This document provides a comprehensive schema and domain reference for all **27 features** present in the RazorGuard AI Payment Fraud Dataset (`data/raw/raw_fraud_transactions.csv` and `data/processed/processed_fraud_transactions.csv`).

---

## 📊 Dataset Metadata Summary
- **Total Features**: 27 (26 inputs + 1 target)
- **Target Feature**: `is_fraud` (Binary: `0` = Legitimate, `1` = Fraudulent)
- **Dataset Type**: Synthetic Payment Network Transactions
- **Class Balance**: ~4.0% Fraud, ~96.0% Legitimate

---

## 📑 Feature Specification Matrix

| Feature Name | Data Type | Description / Domain Meaning | Allowed Range / Values | Available at Prediction? | Target Leakage Risk |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `transaction_id` | String | Unique transaction identifier | `TXN_00000001` - `TXN_99999999` | Yes | None (Identifier) |
| `customer_id` | String | Unique account/customer identifier | `CUST_10000` - `CUST_99999` | Yes | None (Identifier) |
| `merchant_id` | String | Unique merchant identifier | `MERCH_1000` - `MERCH_9999` | Yes | None (Identifier) |
| `amount` | Float | Transaction amount in specified currency | `$1.00` to `$10,000.00+` | Yes | None |
| `currency` | String | ISO 3-letter currency code | `USD`, `EUR`, `GBP`, `INR`, `CAD`, `AUD` | Yes | None |
| `timestamp` | String | Transaction timestamp (ISO 8601) | YYYY-MM-DDTHH:MM:SS | Yes | None |
| `merchant_category` | String | Industry category of merchant | `retail`, `grocery`, `electronics`, `travel`, `digital_goods`, `gaming`, `crypto`, `luxury`, `food_delivery`, `services` | Yes | None |
| `payment_method` | String | Method used to complete transaction | `credit_card`, `debit_card`, `bank_transfer`, `e_wallet` | Yes | None |
| `country` | String | Transaction origin country (ISO 2-letter) | `US`, `GB`, `DE`, `FR`, `IN`, `CA`, `AU`, `JP`, `BR`, `SG` | Yes | None |
| `account_age_days` | Integer | Customer account age in days | `0` to `1800` days | Yes | None |
| `customer_transaction_count` | Integer | Total historical transactions for customer | `1` to `500+` | Yes | None |
| `transactions_last_24h` | Integer | Number of transactions in trailing 24h | `0` to `50+` | Yes | None |
| `transactions_last_7d` | Integer | Number of transactions in trailing 7 days | `0` to `200+` | Yes | None |
| `average_transaction_amount` | Float | Customer's historical mean transaction amount | `$1.00` to `$5,000.00` | Yes | None |
| `payment_attempts` | Integer | Number of payment attempts for this checkout | `1` to `10` | Yes | None |
| `failed_payment_count` | Integer | Number of failed payment attempts for current checkout | `0` to `5` | Yes | None |
| `previous_fraud_count` | Integer | Historical confirmed fraud incidents on account prior to this transaction | `0` to `5` | Yes (Historical) | None (Pre-event) |
| `previous_chargeback_count` | Integer | Historical dispute/chargeback count prior to this transaction | `0` to `5` | Yes (Historical) | None (Pre-event) |
| `unique_devices` | Integer | Number of distinct devices used by customer in last 30 days | `1` to `10` | Yes | None |
| `unique_ips` | Integer | Number of distinct IP addresses used by customer in last 30 days | `1` to `15` | Yes | None |
| `billing_shipping_match` | Integer | Binary flag indicating if billing and shipping address match | `0` (Mismatch), `1` (Match) | Yes | None |
| `country_change` | Integer | Binary flag indicating if IP country differs from home billing country | `0` (Same), `1` (Different) | Yes | None |
| `device_reuse_count` | Integer | Number of accounts sharing this same device | `1` to `100+` | Yes | None |
| `velocity_score` | Float | Calculated transaction rate & frequency risk score | `0.00` to `100.00` | Yes | None |
| `hour_of_day` | Integer | Local transaction hour of day | `0` to `23` | Yes | None |
| `day_of_week` | Integer | Day of week | `0` (Monday) to `6` (Sunday) | Yes | None |
| **`is_fraud`** | Integer | **Target Class Label** | **`0` (Legitimate), `1` (Fraud)** | **No (Ground Truth)** | **Target Variable** |

---

## 🔒 Target Leakage Safeguards
1. **Pre-Event Granularity**: `previous_fraud_count` and `previous_chargeback_count` strictly capture events recorded **before** the timestamp of the current transaction.
2. **No Post-Outcome Information**: Features like authorization outcomes or manual chargeback flags post-transaction are excluded from inference inputs.
3. **No Single Deterministic Feature**: Fraud is modeled as a non-linear combination of velocity, device reuse, location mismatch, amount ratio, and category risk.

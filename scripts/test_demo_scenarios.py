import os
import sys

from backend.app.services.explanation_service import ExplanationService

svc = ExplanationService("models")

s1 = {
    "transaction_id": "TXN_DEMO_SCENARIO_1",
    "customer_id": "CUST_RETURNING_4821",
    "merchant_id": "MERCH_ECOMM_01",
    "amount": 42.50,
    "currency": "USD",
    "merchant_category": "retail",
    "payment_method": "credit_card",
    "country": "US",
    "account_age_days": 365,
    "customer_transaction_count": 48,
    "transactions_last_24h": 1,
    "transactions_last_7d": 4,
    "average_transaction_amount": 45.0,
    "payment_attempts": 1,
    "failed_payment_count": 0,
    "previous_fraud_count": 0,
    "previous_chargeback_count": 0,
    "unique_devices": 1,
    "unique_ips": 1,
    "billing_shipping_match": 1,
    "country_change": 0,
    "device_reuse_count": 1,
    "velocity_score": 5.0,
    "hour_of_day": 14,
    "day_of_week": 2
}

s2 = {
    "transaction_id": "TXN_DEMO_SCENARIO_2",
    "customer_id": "CUST_NEW_9012",
    "merchant_id": "MERCH_ELEC_09",
    "amount": 480.00,
    "currency": "USD",
    "merchant_category": "electronics",
    "payment_method": "credit_card",
    "country": "US",
    "account_age_days": 4,
    "customer_transaction_count": 1,
    "transactions_last_24h": 2,
    "transactions_last_7d": 2,
    "average_transaction_amount": 50.0,
    "payment_attempts": 1,
    "failed_payment_count": 0,
    "previous_fraud_count": 0,
    "previous_chargeback_count": 0,
    "unique_devices": 1,
    "unique_ips": 1,
    "billing_shipping_match": 1,
    "country_change": 0,
    "device_reuse_count": 1,
    "velocity_score": 35.0,
    "hour_of_day": 22,
    "day_of_week": 4
}

s3 = {
    "transaction_id": "TXN_DEMO_SCENARIO_3",
    "customer_id": "CUST_SUSPICIOUS_771",
    "merchant_id": "MERCH_LUX_04",
    "amount": 1250.00,
    "currency": "USD",
    "merchant_category": "luxury",
    "payment_method": "credit_card",
    "country": "US",
    "account_age_days": 12,
    "customer_transaction_count": 8,
    "transactions_last_24h": 9,
    "transactions_last_7d": 15,
    "average_transaction_amount": 85.0,
    "payment_attempts": 4,
    "failed_payment_count": 3,
    "previous_fraud_count": 1,
    "previous_chargeback_count": 1,
    "unique_devices": 5,
    "unique_ips": 4,
    "billing_shipping_match": 0,
    "country_change": 1,
    "device_reuse_count": 8,
    "velocity_score": 88.0,
    "hour_of_day": 3,
    "day_of_week": 6
}

s4 = {
    "transaction_id": "TXN_DEMO_SCENARIO_4",
    "customer_id": "CUST_VIP_1099",
    "merchant_id": "MERCH_TRAVEL_02",
    "amount": 2450.00,
    "currency": "USD",
    "merchant_category": "travel",
    "payment_method": "credit_card",
    "country": "US",
    "account_age_days": 520,
    "customer_transaction_count": 110,
    "transactions_last_24h": 1,
    "transactions_last_7d": 3,
    "average_transaction_amount": 1800.0,
    "payment_attempts": 1,
    "failed_payment_count": 0,
    "previous_fraud_count": 0,
    "previous_chargeback_count": 0,
    "unique_devices": 1,
    "unique_ips": 1,
    "billing_shipping_match": 1,
    "country_change": 0,
    "device_reuse_count": 1,
    "velocity_score": 8.0,
    "hour_of_day": 11,
    "day_of_week": 1
}

for i, s in enumerate([s1, s2, s3, s4], 1):
    res = svc.explain_transaction(s)
    print(f"=== SCENARIO {i} ===")
    print("Fraud Prob:", res["fraud_probability"])
    print("ML Score:", res["ml_risk_score"])
    print("Final Score:", res["final_risk_score"])
    print("Decision:", res["decision"])
    print("Risk Level:", res["risk_level"])
    print("Triggered Rules:", res["triggered_rules"])
    print("Top Risk Factors:", [(f["feature_name_human"], f["shap_value"]) for f in res["risk_factors"][:3]])
    print("Top Protective Factors:", [(f["feature_name_human"], f["shap_value"]) for f in res["protective_factors"][:3]])
    print("LLM Headline:", res["llm_explanation"]["summary_headline"])

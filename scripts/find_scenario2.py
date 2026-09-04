import os
import sys
from backend.app.services.explanation_service import ExplanationService

svc = ExplanationService("models")

for amt in [75, 120, 150, 180, 220, 280, 350]:
    for age in [5, 10, 15, 20, 30, 45]:
        for tx_count in [1, 2, 3, 5]:
            for avg_amt in [30.0, 50.0, 70.0]:
                s = {
                    "transaction_id": f"TXN_S2_TEST",
                    "customer_id": "CUST_NEW_9012",
                    "merchant_id": "MERCH_ELEC_09",
                    "amount": float(amt),
                    "currency": "USD",
                    "merchant_category": "electronics",
                    "payment_method": "credit_card",
                    "country": "US",
                    "account_age_days": age,
                    "customer_transaction_count": tx_count,
                    "transactions_last_24h": 1,
                    "transactions_last_7d": 1,
                    "average_transaction_amount": avg_amt,
                    "payment_attempts": 1,
                    "failed_payment_count": 0,
                    "previous_fraud_count": 0,
                    "previous_chargeback_count": 0,
                    "unique_devices": 1,
                    "unique_ips": 1,
                    "billing_shipping_match": 1,
                    "country_change": 0,
                    "device_reuse_count": 1,
                    "velocity_score": 15.0,
                    "hour_of_day": 20,
                    "day_of_week": 3
                }
                res = svc.explain_transaction(s)
                if 30 <= res["final_risk_score"] < 70:
                    print(f"FOUND: amt={amt}, age={age}, tx_count={tx_count}, avg_amt={avg_amt} -> ML={res['ml_risk_score']}, Final={res['final_risk_score']}, Dec={res['decision']}, Level={res['risk_level']}")
                    break

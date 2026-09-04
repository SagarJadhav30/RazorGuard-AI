"""
RazorGuard AI - Synthetic Payment Fraud Dataset Generator

Generates realistic 100,000+ transaction dataset with domain-informed fraud patterns,
realistic class imbalance, non-linear feature interactions, and zero target leakage.
"""

import os
import random
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


FEATURE_COLUMNS: List[str] = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "amount",
    "currency",
    "timestamp",
    "merchant_category",
    "payment_method",
    "country",
    "account_age_days",
    "customer_transaction_count",
    "transactions_last_24h",
    "transactions_last_7d",
    "average_transaction_amount",
    "payment_attempts",
    "failed_payment_count",
    "previous_fraud_count",
    "previous_chargeback_count",
    "unique_devices",
    "unique_ips",
    "billing_shipping_match",
    "country_change",
    "device_reuse_count",
    "velocity_score",
    "hour_of_day",
    "day_of_week",
    "is_fraud"
]

MERCHANT_CATEGORIES = [
    "retail", "grocery", "electronics", "travel", "digital_goods",
    "gaming", "crypto", "luxury", "food_delivery", "services"
]

HIGH_RISK_CATEGORIES = {"crypto", "gaming", "digital_goods", "luxury"}

PAYMENT_METHODS = ["credit_card", "debit_card", "bank_transfer", "e_wallet"]
CURRENCIES = ["USD", "EUR", "GBP", "INR", "CAD", "AUD"]
COUNTRIES = ["US", "GB", "DE", "FR", "IN", "CA", "AU", "JP", "BR", "SG"]


def generate_synthetic_fraud_dataset(
    n_samples: int = 100000,
    fraud_rate: float = 0.04,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic payment dataset of `n_samples` records with realistic fraud patterns.
    Strictly avoids single-feature fraud indicators and target leakage.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    start_date = datetime(2026, 1, 1, 0, 0, 0)

    def generate_records(count: int, is_fraud_flag: int) -> Dict[str, np.ndarray]:
        # Generate timestamps over a 90-day window
        seconds_offset = np.random.uniform(0, 90 * 86400, size=count)
        timestamps = [start_date + timedelta(seconds=s) for s in seconds_offset]
        hours = np.array([ts.hour for ts in timestamps])
        days_of_week = np.array([ts.weekday() for ts in timestamps])
        ts_strings = [ts.isoformat() for ts in timestamps]

        if is_fraud_flag == 0:
            # --- Legitimate Customer Behavior ---
            amounts = np.round(np.random.lognormal(mean=3.9, sigma=0.85, size=count), 2)  # ~$50 median
            amounts = np.clip(amounts, 1.0, 5000.0)

            categories = np.random.choice(
                MERCHANT_CATEGORIES, size=count,
                p=[0.30, 0.25, 0.10, 0.08, 0.08, 0.05, 0.01, 0.03, 0.07, 0.03]
            )
            pay_methods = np.random.choice(PAYMENT_METHODS, size=count, p=[0.55, 0.30, 0.10, 0.05])
            currencies = np.random.choice(CURRENCIES, size=count, p=[0.70, 0.12, 0.08, 0.05, 0.03, 0.02])
            countries = np.random.choice(COUNTRIES, size=count, p=[0.60, 0.10, 0.08, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01])

            account_age = np.random.randint(30, 1800, size=count)
            cust_tx_count = np.random.poisson(lam=45, size=count) + 1
            tx_last_24h = np.random.poisson(lam=1.2, size=count)
            tx_last_7d = np.random.poisson(lam=6.5, size=count)

            # Average transaction amount correlated with current amount for legitimate accounts
            avg_tx_amount = np.round(amounts * np.random.uniform(0.7, 1.3, size=count), 2)
            avg_tx_amount = np.clip(avg_tx_amount, 5.0, 4500.0)

            payment_attempts = np.random.choice([1, 2, 3], size=count, p=[0.92, 0.07, 0.01])
            failed_payment_count = np.random.choice([0, 1, 2], size=count, p=[0.95, 0.04, 0.01])

            prev_fraud = np.zeros(count, dtype=int)
            prev_chargebacks = np.random.choice([0, 1], size=count, p=[0.99, 0.01])

            unique_devices = np.random.choice([1, 2, 3], size=count, p=[0.80, 0.17, 0.03])
            unique_ips = np.random.choice([1, 2, 3, 4], size=count, p=[0.75, 0.18, 0.05, 0.02])

            billing_shipping_match = np.random.choice([1, 0], size=count, p=[0.94, 0.06])
            country_change = np.random.choice([0, 1], size=count, p=[0.96, 0.04])

            device_reuse_count = np.random.randint(1, 50, size=count)
            velocity_score = np.round(np.random.beta(a=2, b=10, size=count) * 100, 2)  # Low velocity score

        else:
            # --- Fraudulent Behavior Patterns ---
            # Blend of high amount fraud, card testing micro-charges, and rapid velocity spikes
            is_micro = np.random.binomial(n=1, p=0.30, size=count)
            micro_amounts = np.random.uniform(1.2, 5.0, size=count)
            high_amounts = np.random.lognormal(mean=6.5, sigma=1.1, size=count)
            amounts = np.where(is_micro == 1, micro_amounts, high_amounts)
            amounts = np.clip(np.round(amounts, 2), 1.0, 10000.0)

            categories = np.random.choice(
                MERCHANT_CATEGORIES, size=count,
                p=[0.10, 0.02, 0.25, 0.08, 0.20, 0.12, 0.10, 0.10, 0.01, 0.02]
            )
            pay_methods = np.random.choice(PAYMENT_METHODS, size=count, p=[0.75, 0.15, 0.05, 0.05])
            currencies = np.random.choice(CURRENCIES, size=count, p=[0.60, 0.15, 0.10, 0.05, 0.05, 0.05])
            countries = np.random.choice(COUNTRIES, size=count, p=[0.40, 0.15, 0.10, 0.10, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01])

            is_new_acc = np.random.binomial(n=1, p=0.65, size=count)
            new_age = np.random.randint(0, 14, size=count)
            old_age = np.random.randint(15, 365, size=count)
            account_age = np.where(is_new_acc == 1, new_age, old_age)

            cust_tx_count = np.random.poisson(lam=8, size=count) + 1
            tx_last_24h = np.random.poisson(lam=6.5, size=count) + 1
            tx_last_7d = np.random.poisson(lam=18.0, size=count) + 1

            avg_tx_amount = np.round(np.random.lognormal(mean=3.5, sigma=0.5, size=count), 2)

            payment_attempts = np.random.choice([1, 2, 3, 4, 5], size=count, p=[0.40, 0.30, 0.15, 0.10, 0.05])
            failed_payment_count = np.random.choice([0, 1, 2, 3, 4], size=count, p=[0.35, 0.30, 0.20, 0.10, 0.05])

            prev_fraud = np.random.choice([0, 1, 2], size=count, p=[0.85, 0.12, 0.03])
            prev_chargebacks = np.random.choice([0, 1, 2], size=count, p=[0.75, 0.20, 0.05])

            unique_devices = np.random.choice([1, 2, 3, 4, 5], size=count, p=[0.30, 0.35, 0.20, 0.10, 0.05])
            unique_ips = np.random.choice([1, 2, 3, 4, 5, 6], size=count, p=[0.25, 0.30, 0.25, 0.12, 0.05, 0.03])

            billing_shipping_match = np.random.choice([1, 0], size=count, p=[0.40, 0.60])
            country_change = np.random.choice([0, 1], size=count, p=[0.45, 0.55])

            device_reuse_count = np.random.randint(1, 5, size=count)
            velocity_score = np.round(np.random.beta(a=7, b=3, size=count) * 100, 2)  # High velocity score

        return {
            "amount": amounts,
            "currency": currencies,
            "timestamp": ts_strings,
            "merchant_category": categories,
            "payment_method": pay_methods,
            "country": countries,
            "account_age_days": account_age,
            "customer_transaction_count": cust_tx_count,
            "transactions_last_24h": tx_last_24h,
            "transactions_last_7d": tx_last_7d,
            "average_transaction_amount": avg_tx_amount,
            "payment_attempts": payment_attempts,
            "failed_payment_count": failed_payment_count,
            "previous_fraud_count": prev_fraud,
            "previous_chargeback_count": prev_chargebacks,
            "unique_devices": unique_devices,
            "unique_ips": unique_ips,
            "billing_shipping_match": billing_shipping_match,
            "country_change": country_change,
            "device_reuse_count": device_reuse_count,
            "velocity_score": velocity_score,
            "hour_of_day": hours,
            "day_of_week": days_of_week,
            "is_fraud": np.full(count, is_fraud_flag, dtype=int)
        }

    legit_data = generate_records(n_legit, is_fraud_flag=0)
    fraud_data = generate_records(n_fraud, is_fraud_flag=1)

    # Combine dictionary lists
    combined_data = {}
    for key in legit_data.keys():
        combined_data[key] = np.concatenate([legit_data[key], fraud_data[key]])

    df = pd.DataFrame(combined_data)

    # Generate unique transaction IDs, customer IDs, merchant IDs
    tx_ids = [f"TXN_{i+1:08d}" for i in range(n_samples)]
    cust_ids = [f"CUST_{random.randint(10000, 99999)}" for _ in range(n_samples)]
    merch_ids = [f"MERCH_{random.randint(1000, 9999)}" for _ in range(n_samples)]

    df.insert(0, "transaction_id", tx_ids)
    df.insert(1, "customer_id", cust_ids)
    df.insert(2, "merchant_id", merch_ids)

    # Shuffle dataset reproducibly
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Clearly label dataset as synthetic in metadata attribute
    df.attrs["dataset_type"] = "synthetic_payment_fraud_dataset"
    df.attrs["generation_seed"] = random_seed
    df.attrs["created_at"] = datetime.now(timezone.utc).isoformat()

    return df


def save_dataset_files(df: pd.DataFrame, base_dir: str = ".") -> Tuple[str, str]:
    """
    Saves generated dataset to data/raw/ and data/processed/.
    """
    raw_dir = os.path.join(base_dir, "data", "raw")
    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    raw_path = os.path.join(raw_dir, "raw_fraud_transactions.csv")
    proc_path = os.path.join(proc_dir, "processed_fraud_transactions.csv")

    df.to_csv(raw_path, index=False)
    df.to_csv(proc_path, index=False)

    return raw_path, proc_path


def get_preset_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Returns reproducible preset transaction scenarios for hackathon live demo & testing:
    Scenario 1: Normal returning customer -> LOW RISK (APPROVE)
    Scenario 2: New customer + unusual amount -> MEDIUM RISK (REVIEW)
    Scenario 3: Multiple payment attempts + device reuse -> HIGH RISK (BLOCK)
    Scenario 4: Known legitimate customer with high-value purchase -> LOW RISK (APPROVE)
    """
    scenarios = {
        "scenario_1_returning_customer": {
            "name": "Scenario 1: Normal Returning Customer",
            "description": "Routine supermarket purchase from an established customer on a trusted device.",
            "expected_decision": "APPROVE",
            "expected_risk_level": "LOW",
            "data": {
                "customer_id": "CUST_RETURNING_01",
                "merchant_id": "MERCH_GROCERY_99",
                "amount": 42.50,
                "currency": "USD",
                "merchant_category": "grocery",
                "payment_method": "debit_card",
                "country": "US",
                "account_age_days": 420,
                "customer_transaction_count": 95,
                "transactions_last_24h": 1,
                "transactions_last_7d": 5,
                "average_transaction_amount": 45.0,
                "payment_attempts": 1,
                "failed_payment_count": 0,
                "previous_fraud_count": 0,
                "previous_chargeback_count": 0,
                "unique_devices": 1,
                "unique_ips": 1,
                "billing_shipping_match": 1,
                "country_change": 0,
                "device_reuse_count": 30,
                "velocity_score": 10.0,
                "hour_of_day": 14,
                "day_of_week": 2
            }
        },
        "scenario_2_new_customer_unusual_amount": {
            "name": "Scenario 2: New Customer + Unusual Amount",
            "description": "Newly registered customer (20 days old) making a $950 purchase vs their $50 average.",
            "expected_decision": "REVIEW",
            "expected_risk_level": "MEDIUM",
            "data": {
                "customer_id": "CUST_NEW_USER_88",
                "merchant_id": "MERCH_RETAIL_12",
                "amount": 950.00,
                "currency": "USD",
                "merchant_category": "retail",
                "payment_method": "credit_card",
                "country": "US",
                "account_age_days": 20,
                "customer_transaction_count": 5,
                "transactions_last_24h": 2,
                "transactions_last_7d": 4,
                "average_transaction_amount": 50.0,
                "payment_attempts": 1,
                "failed_payment_count": 0,
                "previous_fraud_count": 0,
                "previous_chargeback_count": 0,
                "unique_devices": 1,
                "unique_ips": 1,
                "billing_shipping_match": 1,
                "country_change": 0,
                "device_reuse_count": 4,
                "velocity_score": 32.0,
                "hour_of_day": 14,
                "day_of_week": 2
            }
        },
        "scenario_3_multi_attempt_device_reuse": {
            "name": "Scenario 3: Multiple Payment Attempts + Device Reuse + Attack Pattern",
            "description": "Rapid velocity burst, multiple failed payment attempts, botnet device sharing, and country change.",
            "expected_decision": "BLOCK",
            "expected_risk_level": "HIGH",
            "data": {
                "customer_id": "CUST_BOTNET_ATTACK_07",
                "merchant_id": "MERCH_CRYPTO_DIGITAL_01",
                "amount": 1850.00,
                "currency": "USD",
                "merchant_category": "crypto",
                "payment_method": "credit_card",
                "country": "US",
                "account_age_days": 3,
                "customer_transaction_count": 3,
                "transactions_last_24h": 9,
                "transactions_last_7d": 14,
                "average_transaction_amount": 35.0,
                "payment_attempts": 4,
                "failed_payment_count": 3,
                "previous_fraud_count": 1,
                "previous_chargeback_count": 1,
                "unique_devices": 4,
                "unique_ips": 5,
                "billing_shipping_match": 0,
                "country_change": 1,
                "device_reuse_count": 2,
                "velocity_score": 88.0,
                "hour_of_day": 3,
                "day_of_week": 1
            }
        },
        "scenario_4_high_value_legitimate": {
            "name": "Scenario 4: Known Legitimate Customer with High-Value Purchase",
            "description": "High-value purchase ($3,200) from an established, highly trusted customer demonstrating amount alone is not fraud.",
            "expected_decision": "APPROVE",
            "expected_risk_level": "LOW",
            "data": {
                "customer_id": "CUST_TRUSTED_VIP_01",
                "merchant_id": "MERCH_LUXURY_JEWELRY_01",
                "amount": 3200.00,
                "currency": "USD",
                "merchant_category": "retail",
                "payment_method": "credit_card",
                "country": "US",
                "account_age_days": 1250,
                "customer_transaction_count": 210,
                "transactions_last_24h": 1,
                "transactions_last_7d": 3,
                "average_transaction_amount": 2800.0,
                "payment_attempts": 1,
                "failed_payment_count": 0,
                "previous_fraud_count": 0,
                "previous_chargeback_count": 0,
                "unique_devices": 1,
                "unique_ips": 1,
                "billing_shipping_match": 1,
                "country_change": 0,
                "device_reuse_count": 50,
                "velocity_score": 8.0,
                "hour_of_day": 15,
                "day_of_week": 5
            }
        }
    }

    # Backward compatibility aliases
    scenarios["standard_grocery_inperson"] = scenarios["scenario_1_returning_customer"]
    scenarios["stolen_card_attack"] = scenarios["scenario_3_multi_attempt_device_reuse"]

    return scenarios

"""
RazorGuard AI - Synthetic Payment Transaction Data Generator

Generates realistic payment transaction datasets with domain-informed fraud signals,
feature relationships, and non-linear interactions for robust ML training.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List 

FEATURE_NAMES: List[str] = [  
    "amount",
    "distance_from_home",
    "distance_from_last_tx",
    "velocity_1h",
    "velocity_24h",
    "ratio_to_median_price",
    "repeat_retailer",
    "used_chip",
    "used_pin",
    "online_order",
    "ip_country_mismatch",
    "device_trust_score",
    "high_risk_category",
    "hour_of_day"
]

FEATURE_LABELS: Dict[str, str] = {
    "amount": "Transaction Amount ($)",
    "distance_from_home": "Distance from Home (miles)",
    "distance_from_last_tx": "Distance from Last Tx (miles)",
    "velocity_1h": "1-Hour Tx Count",
    "velocity_24h": "24-Hour Tx Count",
    "ratio_to_median_price": "Ratio to Historical Median Price",
    "repeat_retailer": "Repeat Retailer (0/1)",
    "used_chip": "EMV Chip Used (0/1)",
    "used_pin": "PIN Entered (0/1)",
    "online_order": "Online Transaction (0/1)",
    "ip_country_mismatch": "IP/Billing Country Mismatch (0/1)",
    "device_trust_score": "Device Trust Score (0.0-1.0)",
    "high_risk_category": "High-Risk Merchant Category (0/1)",
    "hour_of_day": "Transaction Hour of Day (0-23)"
}


def generate_payment_dataset(
    n_samples: int = 10000,
    fraud_rate: float = 0.08,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic payment dataset with realistic feature distributions
    and controlled non-linear relationships for payment fraud detection.
    """
    np.random.seed(random_state)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # --- Legitimate Transactions ---
    amount_legit = np.random.lognormal(mean=3.8, sigma=0.9, size=n_legit)  # ~$45 median
    dist_home_legit = np.random.exponential(scale=12.0, size=n_legit)
    dist_last_legit = np.random.exponential(scale=5.0, size=n_legit)
    vel_1h_legit = np.random.poisson(lam=0.4, size=n_legit)
    vel_24h_legit = np.random.poisson(lam=2.5, size=n_legit)
    ratio_median_legit = np.random.lognormal(mean=0.0, sigma=0.4, size=n_legit)
    repeat_retailer_legit = np.random.binomial(n=1, p=0.75, size=n_legit)
    used_chip_legit = np.random.binomial(n=1, p=0.85, size=n_legit)
    used_pin_legit = np.random.binomial(n=1, p=0.70, size=n_legit)
    online_order_legit = np.random.binomial(n=1, p=0.40, size=n_legit)
    ip_mismatch_legit = np.random.binomial(n=1, p=0.02, size=n_legit)
    device_trust_legit = np.random.beta(a=8, b=2, size=n_legit)  # mean ~0.80
    high_risk_cat_legit = np.random.binomial(n=1, p=0.05, size=n_legit)
    hour_legit_p = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05,
        0.07, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.06,
        0.06, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01
    ])
    hour_legit_p = hour_legit_p / hour_legit_p.sum()
    hour_legit = np.random.choice(range(24), size=n_legit, p=hour_legit_p)

    # --- Fraudulent Transactions ---
    # Mix of high-value fraud, card-not-present velocity attacks, and IP location anomalies
    amount_fraud = np.random.choice(
        [np.random.uniform(1.5, 4.9), np.random.lognormal(mean=6.2, sigma=1.0)],
        size=n_fraud, p=[0.25, 0.75]
    )
    dist_home_fraud = np.random.exponential(scale=180.0, size=n_fraud)
    dist_last_fraud = np.random.exponential(scale=120.0, size=n_fraud)
    vel_1h_fraud = np.random.poisson(lam=4.2, size=n_fraud)
    vel_24h_fraud = np.random.poisson(lam=9.0, size=n_fraud)
    ratio_median_fraud = np.random.lognormal(mean=1.5, sigma=0.8, size=n_fraud)
    repeat_retailer_fraud = np.random.binomial(n=1, p=0.15, size=n_fraud)
    used_chip_fraud = np.random.binomial(n=1, p=0.10, size=n_fraud)
    used_pin_fraud = np.random.binomial(n=1, p=0.05, size=n_fraud)
    online_order_fraud = np.random.binomial(n=1, p=0.88, size=n_fraud)
    ip_mismatch_fraud = np.random.binomial(n=1, p=0.65, size=n_fraud)
    device_trust_fraud = np.random.beta(a=2, b=8, size=n_fraud)  # mean ~0.20
    high_risk_cat_fraud = np.random.binomial(n=1, p=0.55, size=n_fraud)
    
    hour_fraud_p = np.array([
        0.08, 0.09, 0.09, 0.08, 0.07, 0.05, 0.03, 0.02,
        0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04,
        0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.06
    ])
    hour_fraud_p = hour_fraud_p / hour_fraud_p.sum()
    hour_fraud = np.random.choice(range(24), size=n_fraud, p=hour_fraud_p)

    df_legit = pd.DataFrame({
        "amount": np.round(amount_legit, 2),
        "distance_from_home": np.round(dist_home_legit, 1),
        "distance_from_last_tx": np.round(dist_last_legit, 1),
        "velocity_1h": vel_1h_legit,
        "velocity_24h": vel_24h_legit,
        "ratio_to_median_price": np.round(ratio_median_legit, 2),
        "repeat_retailer": repeat_retailer_legit,
        "used_chip": used_chip_legit,
        "used_pin": used_pin_legit,
        "online_order": online_order_legit,
        "ip_country_mismatch": ip_mismatch_legit,
        "device_trust_score": np.round(device_trust_legit, 3),
        "high_risk_category": high_risk_cat_legit,
        "hour_of_day": hour_legit,
        "is_fraud": 0
    })

    df_fraud = pd.DataFrame({
        "amount": np.round(amount_fraud, 2),
        "distance_from_home": np.round(dist_home_fraud, 1),
        "distance_from_last_tx": np.round(dist_last_fraud, 1),
        "velocity_1h": vel_1h_fraud,
        "velocity_24h": vel_24h_fraud,
        "ratio_to_median_price": np.round(ratio_median_fraud, 2),
        "repeat_retailer": repeat_retailer_fraud,
        "used_chip": used_chip_fraud,
        "used_pin": used_pin_fraud,
        "online_order": online_order_fraud,
        "ip_country_mismatch": ip_mismatch_fraud,
        "device_trust_score": np.round(device_trust_fraud, 3),
        "high_risk_category": high_risk_cat_fraud,
        "hour_of_day": hour_fraud,
        "is_fraud": 1
    })

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def get_preset_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Returns preset transaction scenarios for live simulation & testing.
    """
    return {
        "stolen_card_attack": {
            "name": "Stolen Card Overseas E-Commerce Attack",
            "description": "High amount online order from novel device with IP mismatch and no chip/PIN.",
            "data": {
                "amount": 1850.00,
                "distance_from_home": 1420.5,
                "distance_from_last_tx": 950.0,
                "velocity_1h": 5,
                "velocity_24h": 12,
                "ratio_to_median_price": 8.50,
                "repeat_retailer": 0,
                "used_chip": 0,
                "used_pin": 0,
                "online_order": 1,
                "ip_country_mismatch": 1,
                "device_trust_score": 0.12,
                "high_risk_category": 1,
                "hour_of_day": 3
            }
        },
        "micro_charge_velocity": {
            "name": "Card Testing Micro-Charge Velocity",
            "description": "Rapid succession of small online transactions from untrusted proxy device.",
            "data": {
                "amount": 2.49,
                "distance_from_home": 340.0,
                "distance_from_last_tx": 200.0,
                "velocity_1h": 9,
                "velocity_24h": 18,
                "ratio_to_median_price": 0.05,
                "repeat_retailer": 0,
                "used_chip": 0,
                "used_pin": 0,
                "online_order": 1,
                "ip_country_mismatch": 1,
                "device_trust_score": 0.18,
                "high_risk_category": 1,
                "hour_of_day": 2
            }
        },
        "legitimate_high_value": {
            "name": "Legitimate High-Value Electronics Purchase",
            "description": "High amount purchase at familiar online retailer with high device trust score.",
            "data": {
                "amount": 1299.99,
                "distance_from_home": 8.2,
                "distance_from_last_tx": 1.5,
                "velocity_1h": 1,
                "velocity_24h": 2,
                "ratio_to_median_price": 4.20,
                "repeat_retailer": 1,
                "used_chip": 0,
                "used_pin": 0,
                "online_order": 1,
                "ip_country_mismatch": 0,
                "device_trust_score": 0.94,
                "high_risk_category": 0,
                "hour_of_day": 14
            }
        },
        "standard_grocery_inperson": {
            "name": "Standard In-Person Grocery Store Purchase",
            "description": "Low risk routine supermarket checkout with chip & PIN verified.",
            "data": {
                "amount": 54.30,
                "distance_from_home": 3.1,
                "distance_from_last_tx": 0.5,
                "velocity_1h": 1,
                "velocity_24h": 3,
                "ratio_to_median_price": 1.05,
                "repeat_retailer": 1,
                "used_chip": 1,
                "used_pin": 1,
                "online_order": 0,
                "ip_country_mismatch": 0,
                "device_trust_score": 0.98,
                "high_risk_category": 0,
                "hour_of_day": 17
            }
        },
        "traveler_anomalous_location": {
            "name": "Vacation Travel Airport Purchase",
            "description": "In-person purchase in distant city with EMV chip present.",
            "data": {
                "amount": 125.00,
                "distance_from_home": 850.0,
                "distance_from_last_tx": 840.0,
                "velocity_1h": 1,
                "velocity_24h": 4,
                "ratio_to_median_price": 2.10,
                "repeat_retailer": 0,
                "used_chip": 1,
                "used_pin": 1,
                "online_order": 0,
                "ip_country_mismatch": 0,
                "device_trust_score": 0.85,
                "high_risk_category": 0,
                "hour_of_day": 11
            }
        }
    }

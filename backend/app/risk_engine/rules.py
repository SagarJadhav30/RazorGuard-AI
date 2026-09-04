"""
RazorGuard AI - Deterministic Compliance Policy Rules

Defines hard policy overrides that execute AFTER or IN TANDEM with ML probabilities.
These rules enforce strict merchant safety policies and compliance overrides.
"""

from typing import Dict, Any, List, Tuple

HIGH_RISK_COUNTRIES = {"XX", "ZZ", "SY", "KP"}


def evaluate_deterministic_rules(
    transaction: Dict[str, Any],
    initial_score: int
) -> Tuple[int, List[str]]:
    """
    Applies deterministic compliance rules to initial ML risk score.
    Returns (adjusted_risk_score, triggered_rule_codes).
    """
    score = initial_score
    triggered_rules: List[str] = []

    # 1. Rule: Blacklisted / High-Risk Sanctioned Location Override
    country = str(transaction.get("country", "")).upper()
    if country in HIGH_RISK_COUNTRIES:
        score = 100
        triggered_rules.append("RULE_SANCTIONED_COUNTRY_BLOCK")

    # 2. Rule: Brute-Force Checkout Payment Retries
    failed_attempts = int(transaction.get("failed_payment_count", 0))
    if failed_attempts >= 3:
        score = max(score, 80)
        triggered_rules.append("RULE_BRUTE_FORCE_AUTH_BLOCK")

    # 3. Rule: Velocity Surge Attack
    tx_24h = int(transaction.get("transactions_last_24h", 0))
    vel_score = float(transaction.get("velocity_score", 0.0))
    if tx_24h >= 8 or vel_score >= 80.0:
        score = max(score, 75)
        triggered_rules.append("RULE_HIGH_VELOCITY_SURGE_BLOCK")
    elif tx_24h >= 5 or vel_score >= 60.0:
        score = max(score, 50)
        triggered_rules.append("RULE_MODERATE_VELOCITY_REVIEW")

    # 4. Rule: High-Value Anomaly Mismatch
    amount = float(transaction.get("amount", 0.0))
    country_change = int(transaction.get("country_change", 0))
    billing_match = int(transaction.get("billing_shipping_match", 1))
    if amount >= 1500.0 and country_change == 1 and billing_match == 0:
        score = max(score, 85)
        triggered_rules.append("RULE_HIGH_VALUE_LOCATION_MISMATCH_BLOCK")

    # 5. Rule: Trusted Loyalty Account Ceiling
    account_age = int(transaction.get("account_age_days", 0))
    prev_fraud = int(transaction.get("previous_fraud_count", 0))
    device_reuse = int(transaction.get("device_reuse_count", 1))
    if (
        account_age > 180 and
        prev_fraud == 0 and
        billing_match == 1 and
        country_change == 0 and
        device_reuse <= 3 and
        failed_attempts == 0 and
        country not in HIGH_RISK_COUNTRIES and
        amount < 1000.0
    ):
        score = min(score, 25)
        triggered_rules.append("RULE_TRUSTED_LOYALTY_CAP_APPROVE")

    # Clamp score to [0, 100]
    final_score = max(0, min(100, score))
    return final_score, triggered_rules

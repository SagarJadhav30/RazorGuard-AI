"""
RazorGuard AI - Risk Scoring & Level Mapping

Converts calibrated ML fraud probability into a 0-100 Risk Score,
and maps scores to Risk Levels (LOW, MEDIUM, HIGH) based on configurable settings.
"""

from typing import Dict, Any, Tuple
from backend.app.core.config import settings


def probability_to_risk_score(probability: float) -> int:
    """
    Converts ML fraud probability P in [0.0, 1.0] to an integer Risk Score S in [0, 100].
    """
    prob_clamped = max(0.0, min(1.0, float(probability)))
    return int(round(prob_clamped * 100.0))


def map_score_to_risk_level_and_decision(
    risk_score: int,
    low_threshold: int = 30,
    high_threshold: int = 70
) -> Tuple[str, str]:
    """
    Maps Risk Score (0-100) to Risk Level (LOW, MEDIUM, HIGH) and Decision (APPROVE, REVIEW, BLOCK).

    Rules:
    - Score < low_threshold (e.g., < 30): LOW risk -> APPROVE
    - low_threshold <= Score < high_threshold (e.g., 30..69): MEDIUM risk -> REVIEW
    - Score >= high_threshold (e.g., >= 70): HIGH risk -> BLOCK
    """
    score = int(max(0, min(100, risk_score)))

    if score < low_threshold:
        return "LOW", "APPROVE"
    elif score < high_threshold:
        return "MEDIUM", "REVIEW"
    else:
        return "HIGH", "BLOCK"

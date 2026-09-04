"""
RazorGuard AI - Pydantic Validation & API Response Schemas
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator


class TransactionAssessmentRequest(BaseModel):
    transaction_id: Optional[str] = Field(default=None, description="Optional custom transaction ID. Auto-generated if omitted.")
    customer_id: str = Field(..., min_length=1, max_length=64, description="Unique customer identifier", json_schema_extra={"example": "CUST_99124"})
    merchant_id: str = Field(..., min_length=1, max_length=64, description="Unique merchant identifier", json_schema_extra={"example": "MERCH_4021"})
    amount: float = Field(..., gt=0.0, description="Transaction amount in specified currency (must be > 0)")
    currency: str = Field(default="USD", min_length=3, max_length=3, pattern=r"^[A-Za-z]{3}$", description="3-letter ISO currency code")
    merchant_category: str = Field(..., description="Industry category of merchant")
    payment_method: str = Field(..., description="Checkout payment method")
    country: str = Field(..., description="Origin country code")

    account_age_days: int = Field(default=30, ge=0, description="Customer account age in days")
    customer_transaction_count: int = Field(default=1, ge=1, description="Total historical customer transactions")
    transactions_last_24h: int = Field(default=1, ge=0, description="Trailing 24h transaction volume")
    transactions_last_7d: int = Field(default=4, ge=0, description="Trailing 7d transaction volume")
    average_transaction_amount: float = Field(default=50.0, ge=0.0, description="Customer historical mean amount")
    payment_attempts: int = Field(default=1, ge=1, description="Checkout payment attempt count")
    failed_payment_count: int = Field(default=0, ge=0, description="Failed payment attempt count")
    previous_fraud_count: int = Field(default=0, ge=0, description="Historical confirmed fraud count")
    previous_chargeback_count: int = Field(default=0, ge=0, description="Historical chargeback count")
    unique_devices: int = Field(default=1, ge=1, description="Unique devices in last 30 days")
    unique_ips: int = Field(default=1, ge=1, description="Unique IP addresses in last 30 days")
    billing_shipping_match: int = Field(default=1, description="1 if match, 0 if mismatch")
    country_change: int = Field(default=0, description="1 if IP country differs from billing country")
    device_reuse_count: int = Field(default=1, ge=1, description="Accounts sharing device")
    velocity_score: float = Field(default=10.0, ge=0.0, description="Velocity risk score")
    hour_of_day: int = Field(default=14, ge=0, le=23, description="Local transaction hour of day (0-23)")
    day_of_week: int = Field(default=2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")

    customer_notes: Optional[str] = Field(default=None, description="Freeform customer checkout notes (untrusted data)")


class SHAPFactor(BaseModel):
    feature: str
    feature_name_human: str
    feature_value: float
    shap_value: float
    impact_direction: str


class LLMExplanation(BaseModel):
    summary_headline: str
    why_flagged: str
    top_risk_factors: List[str]
    recommended_analyst_action: str
    confidence_and_limitations: str
    disclaimer: str
    provider_used: str


class TransactionAssessmentResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    ml_risk_score: int
    final_risk_score: int
    anomaly_score: float
    risk_level: str
    decision: str
    triggered_rules: List[str]
    fallback_active: bool
    risk_factors: List[SHAPFactor]
    protective_factors: List[SHAPFactor]
    llm_explanation: LLMExplanation
    evaluation_timestamp: datetime


class TransactionListItem(BaseModel):
    transaction_id: str
    customer_id: str
    merchant_id: str
    amount: float
    currency: str
    merchant_category: str
    country: str
    fraud_probability: float
    final_risk_score: int
    risk_level: str
    decision: str
    created_at: datetime


class TransactionListResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    items: List[TransactionListItem]


class TransactionAssessmentDetail(BaseModel):
    fraud_probability: float
    ml_risk_score: int
    final_risk_score: int
    anomaly_score: float
    risk_level: str
    decision: str
    triggered_rules: List[str]
    fallback_active: bool


class TransactionDetailResponse(BaseModel):
    transaction_id: str
    customer_id: str
    merchant_id: str
    amount: float
    currency: str
    merchant_category: str
    payment_method: str
    country: str
    created_at: datetime
    raw_payload: Dict[str, Any]
    assessment: TransactionAssessmentDetail
    risk_factors: List[SHAPFactor]
    protective_factors: List[SHAPFactor]
    llm_explanation: Dict[str, Any]


class RiskSummaryResponse(BaseModel):
    total_transactions_assessed: int
    approved_count: int
    review_count: int
    blocked_count: int
    approval_rate_pct: float
    review_rate_pct: float
    block_rate_pct: float
    total_volume_processed_usd: float
    high_risk_volume_blocked_usd: float


class ModelMetricsResponse(BaseModel):
    model_type: str
    n_heldout_test_samples: int
    metrics: Dict[str, float]
    confusion_matrix: Dict[str, int]
    cost_analysis: Dict[str, Any]
    feature_names: List[str]
    roc_curve_data: List[Dict[str, Any]]
    pr_curve_data: List[Dict[str, Any]]


class AuditLogResponse(BaseModel):
    id: int
    request_id: str
    event_type: str
    transaction_id: Optional[str]
    model_version: str
    fraud_probability: Optional[float]
    action_taken: str
    action: str
    actor: str
    reason: str
    top_risk_factors: List[Union[Dict[str, Any], str]]
    risk_score: Optional[int]
    decision: Optional[str]
    performed_by: str
    details: Dict[str, Any]
    timestamp: datetime


class AuditLogListResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    items: List[AuditLogResponse]


class VerificationRequest(BaseModel):
    analyst_action: Literal["CONFIRM_FRAUD", "CONFIRM_LEGIT", "OVERRIDE_APPROVE", "OVERRIDE_BLOCK"] = Field(..., json_schema_extra={"example": "CONFIRM_FRAUD"}, description="Analyst verification action")
    analyst_notes: Optional[str] = Field(default=None, description="Analyst review notes and rationale")
    analyst_id: Optional[str] = Field(default="ANALYST_SEC_01", description="Analyst employee ID")


class VerificationResponse(BaseModel):
    transaction_id: str
    previous_decision: str
    updated_decision: str
    analyst_action: str
    verification_id: int
    recorded_at: datetime

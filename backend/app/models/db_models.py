"""
RazorGuard AI - SQLAlchemy Database Models

Defines ORM models for transactions, risk assessments, SHAP explanations,
immutable audit logs, and analyst verification actions.
"""

from datetime import datetime, timezone
import json
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, ForeignKey, Boolean, Index
from sqlalchemy.orm import relationship

from backend.app.database.session import Base


class TransactionModel(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String(64), unique=True, index=True, nullable=False)
    customer_id = Column(String(64), index=True, nullable=False)
    merchant_id = Column(String(64), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    merchant_category = Column(String(50), nullable=False)
    payment_method = Column(String(50), nullable=False)
    country = Column(String(10), nullable=False)

    # Risk signals & attributes stored as JSON string
    raw_payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Relationships
    assessment = relationship("RiskAssessmentModel", back_populates="transaction", uselist=False, cascade="all, delete-orphan")
    verifications = relationship("VerificationModel", back_populates="transaction", cascade="all, delete-orphan")


class RiskAssessmentModel(Base):
    __tablename__ = "risk_predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String(64), ForeignKey("transactions.transaction_id"), unique=True, index=True, nullable=False)
    model_version = Column(String(64), index=True, nullable=False, default="risk-model-v1")

    fraud_probability = Column(Float, nullable=False)
    ml_risk_score = Column(Integer, nullable=False)
    final_risk_score = Column(Integer, nullable=False)
    anomaly_score = Column(Float, default=0.0)
    risk_level = Column(String(20), nullable=False, index=True)  # LOW, MEDIUM, HIGH
    decision = Column(String(20), nullable=False, index=True)    # APPROVE, REVIEW, BLOCK

    triggered_rules_json = Column(Text, default="[]")
    fallback_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    transaction = relationship("TransactionModel", back_populates="assessment")
    risk_factors = relationship("ShapExplanationModel", back_populates="prediction", cascade="all, delete-orphan")


RiskPredictionModel = RiskAssessmentModel


class ShapExplanationModel(Base):
    __tablename__ = "risk_factors"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String(64), ForeignKey("risk_predictions.transaction_id"), unique=True, index=True, nullable=False)

    risk_factors_json = Column(Text, nullable=False)      # JSON list of top risk factors
    protective_factors_json = Column(Text, nullable=False)  # JSON list of top protective factors
    llm_explanation_json = Column(Text, nullable=False)   # JSON LLM risk analyst narrative

    # Relationship
    prediction = relationship("RiskAssessmentModel", back_populates="risk_factors")


RiskFactorModel = ShapExplanationModel


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String(64), index=True, nullable=False)
    event_type = Column(String(50), index=True, nullable=False)  # TRANSACTION_ASSESSED, VERIFICATION_UPDATED
    transaction_id = Column(String(64), index=True, nullable=True)
    model_version = Column(String(64), index=True, nullable=False, default="risk-model-v1")
    fraud_probability = Column(Float, nullable=True)

    action_taken = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False, default="ASSESS")
    actor = Column(String(64), nullable=False, default="SYSTEM_RISK_ENGINE")
    reason = Column(Text, nullable=False, default="Automated risk assessment")
    top_risk_factors = Column(Text, nullable=False, default="[]")
    risk_score = Column(Integer, nullable=True)
    decision = Column(String(20), nullable=True)
    performed_by = Column(String(50), default="SYSTEM_RISK_ENGINE")
    details_json = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_audit_logs_transaction_timestamp", "transaction_id", "timestamp"),
        Index("ix_audit_logs_decision_timestamp", "decision", "timestamp"),
    )


class VerificationModel(Base):
    __tablename__ = "verification_actions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String(64), ForeignKey("transactions.transaction_id"), index=True, nullable=False)

    previous_decision = Column(String(20), nullable=False)
    analyst_action = Column(String(50), nullable=False)  # CONFIRM_FRAUD, CONFIRM_LEGIT, OVERRIDE_APPROVE, OVERRIDE_BLOCK
    analyst_notes = Column(Text, nullable=True)
    analyst_id = Column(String(64), default="ANALYST_DEFAULT")

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship
    transaction = relationship("TransactionModel", back_populates="verifications")


VerificationActionModel = VerificationModel


class ModelVersionModel(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    version = Column(String(64), unique=True, index=True, nullable=False)
    model_type = Column(String(100), nullable=False)
    artifact_fingerprint = Column(String(128), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

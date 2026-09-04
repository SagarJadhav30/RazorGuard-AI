"""
RazorGuard AI - Risk Prediction, Analytics & Model Metrics Endpoints
"""

import json
import logging
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.session import get_db
from backend.app.core.config import settings
from backend.app.schemas.risk import (
    TransactionAssessmentRequest,
    TransactionAssessmentResponse,
    RiskSummaryResponse,
    ModelMetricsResponse,
    TransactionListResponse,
    TransactionDetailResponse,
)
from backend.app.models.db_models import (
    TransactionModel, RiskAssessmentModel, ShapExplanationModel, AuditLogModel, ModelVersionModel
)
from backend.app.services.explanation_service import ExplanationService
from backend.ml.trainer import load_metrics_summary

router = APIRouter()
logger = logging.getLogger("razorguard.api.risk")
explanation_service = ExplanationService(models_dir="models")


@router.post("/risk/predict", response_model=TransactionAssessmentResponse, summary="Assess Payment Transaction Risk")
def predict_transaction_risk(
    req_body: TransactionAssessmentRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Real-time assessment flow:
    Request -> Validation -> Preprocessing -> ML Prediction -> Risk Score -> Deterministic Decision -> SHAP Explanation -> GenAI Explanation -> DB Record -> Audit Log -> Response.
    """
    request_id = getattr(request.state, "request_id", "REQ_INTERNAL")

    # Generate transaction ID if not provided
    tx_id = req_body.transaction_id or f"TXN_{random.randint(10000000, 99999999)}"

    # Build raw dictionary
    tx_dict = req_body.model_dump()
    tx_dict["transaction_id"] = tx_id

    # 1. Execute Unified Explanation Service (ML + Rules + SHAP + LLM)
    try:
        exp_result = explanation_service.explain_transaction(tx_dict)
    except Exception as e:
        logger.exception("Risk engine assessment failed", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="Risk assessment is temporarily unavailable.") from e

    # 2. Persist to Database
    try:
        # Check if transaction ID already exists
        existing = db.query(TransactionModel).filter(TransactionModel.transaction_id == tx_id).first()
        if not existing or not existing.assessment:
            model_version = db.query(ModelVersionModel).filter(ModelVersionModel.version == settings.MODEL_VERSION).first()
            if not model_version:
                db.add(ModelVersionModel(
                    version=settings.MODEL_VERSION,
                    model_type="Hybrid ML + deterministic rules",
                    is_active=True,
                ))
            if not existing:
                tx_record = TransactionModel(
                    transaction_id=tx_id,
                    customer_id=req_body.customer_id,
                    merchant_id=req_body.merchant_id,
                    amount=req_body.amount,
                    currency=req_body.currency,
                    merchant_category=req_body.merchant_category,
                    payment_method=req_body.payment_method,
                    country=req_body.country,
                    raw_payload_json=json.dumps(tx_dict)
                )
                db.add(tx_record)
                db.flush()

            assessment_record = RiskAssessmentModel(
                transaction_id=tx_id,
                model_version=settings.MODEL_VERSION,
                fraud_probability=exp_result["fraud_probability"],
                ml_risk_score=exp_result["ml_risk_score"],
                final_risk_score=exp_result["final_risk_score"],
                anomaly_score=exp_result["anomaly_score"],
                risk_level=exp_result["risk_level"],
                decision=exp_result["decision"],
                triggered_rules_json=json.dumps(exp_result["triggered_rules"]),
                fallback_active=exp_result["fallback_active"]
            )
            db.add(assessment_record)
            db.flush()

            shap_record = ShapExplanationModel(
                transaction_id=tx_id,
                risk_factors_json=json.dumps(exp_result["risk_factors"]),
                protective_factors_json=json.dumps(exp_result["protective_factors"]),
                llm_explanation_json=json.dumps(exp_result["llm_explanation"])
            )
            db.add(shap_record)

            # Audit Log Entry
            audit_entry = AuditLogModel(
                request_id=request_id,
                event_type="TRANSACTION_ASSESSED",
                transaction_id=tx_id,
                model_version=settings.MODEL_VERSION,
                fraud_probability=exp_result["fraud_probability"],
                action_taken=exp_result["decision"],
                action="ASSESS",
                actor="HYBRID_RISK_ENGINE",
                reason="; ".join(exp_result["triggered_rules"]) or "No deterministic rules triggered",
                top_risk_factors=json.dumps(exp_result["risk_factors"][:5]),
                risk_score=exp_result["final_risk_score"],
                decision=exp_result["decision"],
                performed_by="HYBRID_RISK_ENGINE",
                details_json=json.dumps({
                    "amount": req_body.amount,
                    "currency": req_body.currency,
                    "risk_level": exp_result["risk_level"],
                    "fraud_probability": exp_result["fraud_probability"]
                })
            )
            db.add(audit_entry)
            db.commit()

    except Exception as e:
        db.rollback()
        logger.exception("Failed to persist transaction assessment", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="Risk assessment completed, but the result could not be persisted.") from e

    exp_result["transaction_id"] = tx_id
    return exp_result


@router.get("/risk/summary", response_model=RiskSummaryResponse, summary="Get Risk Analytics Summary")
def get_risk_summary(db: Session = Depends(get_db)):
    """
    Returns aggregate risk analytics and decision statistics.
    """
    try:
        total_tx = db.query(func.count(RiskAssessmentModel.id)).scalar() or 0
        approved = db.query(func.count(RiskAssessmentModel.id)).filter(RiskAssessmentModel.decision == "APPROVE").scalar() or 0
        review = db.query(func.count(RiskAssessmentModel.id)).filter(RiskAssessmentModel.decision == "REVIEW").scalar() or 0
        blocked = db.query(func.count(RiskAssessmentModel.id)).filter(RiskAssessmentModel.decision == "BLOCK").scalar() or 0
        total_vol = db.query(func.sum(TransactionModel.amount)).scalar() or 0.0
        blocked_vol = db.query(func.sum(TransactionModel.amount)).join(RiskAssessmentModel).filter(RiskAssessmentModel.decision == "BLOCK").scalar() or 0.0
    except Exception as e:
        logger.exception("Risk summary database query failed")
        raise HTTPException(status_code=503, detail="Risk summary is temporarily unavailable.") from e

    app_rate = round((approved / total_tx) * 100.0, 2) if total_tx > 0 else 0.0
    rev_rate = round((review / total_tx) * 100.0, 2) if total_tx > 0 else 0.0
    blk_rate = round((blocked / total_tx) * 100.0, 2) if total_tx > 0 else 0.0

    return RiskSummaryResponse(
        total_transactions_assessed=total_tx,
        approved_count=approved,
        review_count=review,
        blocked_count=blocked,
        approval_rate_pct=app_rate,
        review_rate_pct=rev_rate,
        block_rate_pct=blk_rate,
        total_volume_processed_usd=round(total_vol, 2),
        high_risk_volume_blocked_usd=round(blocked_vol, 2)
    )


@router.get("/model/metrics", response_model=ModelMetricsResponse, summary="Get Model Held-Out Evaluation Metrics")
def get_model_metrics():
    """
    Returns non-fake held-out test set metrics calculated strictly from actual dataset predictions.
    """
    try:
        metrics_dict = load_metrics_summary()
        return metrics_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model metrics: {str(e)}")

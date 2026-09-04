"""
RazorGuard AI - Transactions Search & Analyst Verification Endpoints
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.core.config import settings
from backend.app.schemas.risk import (
    VerificationRequest,
    VerificationResponse,
    TransactionListResponse,
    TransactionDetailResponse,
)
from backend.app.models.db_models import (
    TransactionModel, RiskAssessmentModel, ShapExplanationModel, AuditLogModel, VerificationModel
)

router = APIRouter()


@router.get("/transactions", response_model=TransactionListResponse, summary="List & Search Evaluated Transactions")
def list_transactions(
    decision: Optional[str] = Query(None, description="Filter by decision: APPROVE, REVIEW, BLOCK"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level: LOW, MEDIUM, HIGH"),
    customer_id: Optional[str] = Query(None, description="Filter by customer ID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Search and filter evaluated payment transactions.
    """
    query = db.query(
        TransactionModel, RiskAssessmentModel
    ).join(
        RiskAssessmentModel, TransactionModel.transaction_id == RiskAssessmentModel.transaction_id
    )

    if decision:
        query = query.filter(RiskAssessmentModel.decision == decision.upper())
    if risk_level:
        query = query.filter(RiskAssessmentModel.risk_level == risk_level.upper())
    if customer_id:
        query = query.filter(TransactionModel.customer_id == customer_id)

    total_count = query.count()
    results = query.order_by(TransactionModel.created_at.desc()).offset(offset).limit(limit).all()

    items = []
    for tx, ra in results:
        items.append({
            "transaction_id": tx.transaction_id,
            "customer_id": tx.customer_id,
            "merchant_id": tx.merchant_id,
            "amount": tx.amount,
            "currency": tx.currency,
            "merchant_category": tx.merchant_category,
            "country": tx.country,
            "fraud_probability": ra.fraud_probability,
            "final_risk_score": ra.final_risk_score,
            "risk_level": ra.risk_level,
            "decision": ra.decision,
            "created_at": tx.created_at
        })

    return {
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "items": items
    }


@router.get("/transactions/{transaction_id}", response_model=TransactionDetailResponse, summary="Get Detailed Transaction & SHAP Breakdown")
def get_transaction_detail(
    transaction_id: str,
    db: Session = Depends(get_db)
):
    """
    Returns full transaction metadata, risk score, SHAP factor breakdown, and LLM explanation.
    """
    tx = db.query(TransactionModel).filter(TransactionModel.transaction_id == transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail=f"Transaction '{transaction_id}' not found.")

    ra = db.query(RiskAssessmentModel).filter(RiskAssessmentModel.transaction_id == transaction_id).first()
    se = db.query(ShapExplanationModel).filter(ShapExplanationModel.transaction_id == transaction_id).first()

    risk_factors = json.loads(se.risk_factors_json) if se else []
    protective_factors = json.loads(se.protective_factors_json) if se else []
    llm_exp = json.loads(se.llm_explanation_json) if se else {}
    triggered_rules = json.loads(ra.triggered_rules_json) if ra else []
    raw_payload = json.loads(tx.raw_payload_json) if tx.raw_payload_json else {}

    return {
        "transaction_id": tx.transaction_id,
        "customer_id": tx.customer_id,
        "merchant_id": tx.merchant_id,
        "amount": tx.amount,
        "currency": tx.currency,
        "merchant_category": tx.merchant_category,
        "payment_method": tx.payment_method,
        "country": tx.country,
        "created_at": tx.created_at,
        "raw_payload": raw_payload,
        "assessment": {
            "fraud_probability": ra.fraud_probability if ra else 0.0,
            "ml_risk_score": ra.ml_risk_score if ra else 0,
            "final_risk_score": ra.final_risk_score if ra else 0,
            "anomaly_score": ra.anomaly_score if ra else 0.0,
            "risk_level": ra.risk_level if ra else "LOW",
            "decision": ra.decision if ra else "APPROVE",
            "triggered_rules": triggered_rules,
            "fallback_active": ra.fallback_active if ra else False
        },
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "llm_explanation": llm_exp
    }


@router.post("/verification/{transaction_id}", response_model=VerificationResponse, summary="Submit Analyst Verification & Decision Override")
def submit_verification(
    transaction_id: str,
    req: VerificationRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Submits security analyst verification action (e.g. CONFIRM_FRAUD, CONFIRM_LEGIT, OVERRIDE_APPROVE, OVERRIDE_BLOCK).
    Updates database status and logs immutable audit trail entry.
    """
    request_id = getattr(request.state, "request_id", "REQ_INTERNAL")

    ra = db.query(RiskAssessmentModel).filter(RiskAssessmentModel.transaction_id == transaction_id).first()
    if not ra:
        raise HTTPException(status_code=404, detail=f"Transaction '{transaction_id}' not found.")

    prev_decision = ra.decision

    # Determine updated decision based on analyst action
    updated_decision = prev_decision
    if req.analyst_action in ["CONFIRM_FRAUD", "OVERRIDE_BLOCK"]:
        updated_decision = "BLOCK"
        ra.decision = "BLOCK"
        ra.risk_level = "HIGH"
    elif req.analyst_action in ["CONFIRM_LEGIT", "OVERRIDE_APPROVE"]:
        updated_decision = "APPROVE"
        ra.decision = "APPROVE"
        ra.risk_level = "LOW"

    v_record = VerificationModel(
        transaction_id=transaction_id,
        previous_decision=prev_decision,
        analyst_action=req.analyst_action,
        analyst_notes=req.analyst_notes,
        analyst_id=req.analyst_id or "ANALYST_SEC_01"
    )
    db.add(v_record)
    db.flush()

    # Immutable Audit Trail Entry
    audit_entry = AuditLogModel(
        request_id=request_id,
        event_type="VERIFICATION_UPDATED",
        transaction_id=transaction_id,
        model_version=settings.MODEL_VERSION,
        fraud_probability=0.0,
        action_taken=req.analyst_action,
        action="VERIFY",
        actor=req.analyst_id or "ANALYST_SEC_01",
        reason=req.analyst_notes or "Analyst verification action",
        top_risk_factors="[]",
        risk_score=ra.final_risk_score,
        decision=updated_decision,
        performed_by=req.analyst_id or "ANALYST_SEC_01",
        details_json=json.dumps({
            "previous_decision": prev_decision,
            "updated_decision": updated_decision,
            "notes": req.analyst_notes
        })
    )
    db.add(audit_entry)
    db.commit()

    return VerificationResponse(
        transaction_id=transaction_id,
        previous_decision=prev_decision,
        updated_decision=updated_decision,
        analyst_action=req.analyst_action,
        verification_id=v_record.id,
        recorded_at=v_record.created_at
    )

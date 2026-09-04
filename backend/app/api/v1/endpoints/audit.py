"""
RazorGuard AI - Immutable Audit Log Endpoints
"""

import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.risk import AuditLogListResponse
from backend.app.models.db_models import AuditLogModel

router = APIRouter()


@router.get("/audit", response_model=AuditLogListResponse, summary="Get Immutable Audit Trail Logs")
def get_audit_logs(
    event_type: Optional[str] = Query(None, description="Filter by event type (e.g. TRANSACTION_ASSESSED, VERIFICATION_UPDATED)"),
    transaction_id: Optional[str] = Query(None, description="Filter by transaction ID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Returns immutable audit log records for compliance and forensic inspection.
    """
    query = db.query(AuditLogModel)

    if event_type:
        query = query.filter(AuditLogModel.event_type == event_type.upper())
    if transaction_id:
        query = query.filter(AuditLogModel.transaction_id == transaction_id)

    total_count = query.count()
    records = query.order_by(AuditLogModel.timestamp.desc()).offset(offset).limit(limit).all()

    items = []
    for r in records:
        details = json.loads(r.details_json) if r.details_json else {}
        items.append({
            "id": r.id,
            "request_id": r.request_id,
            "event_type": r.event_type,
            "transaction_id": r.transaction_id,
            "model_version": r.model_version,
            "fraud_probability": r.fraud_probability,
            "action_taken": r.action_taken,
            "action": r.action,
            "actor": r.actor,
            "reason": r.reason,
            "top_risk_factors": json.loads(r.top_risk_factors) if r.top_risk_factors else [],
            "risk_score": r.risk_score,
            "decision": r.decision,
            "performed_by": r.performed_by,
            "details": details,
            "timestamp": r.timestamp
        })

    return {
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "items": items
    }

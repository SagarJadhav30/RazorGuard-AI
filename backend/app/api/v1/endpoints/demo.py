"""
RazorGuard AI - Hackathon Demo Controller Endpoints

Provides dedicated endpoints to:
1. List reproducible demo scenarios
2. Execute a controlled demo scenario with end-to-end explainability (ML + Rules + SHAP + LLM + DB + Audit)
3. Reset demo state to a clean baseline for repeatable live presentations
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.core.config import settings
from backend.app.demo.scenarios import (
    DEMO_SCENARIOS,
    get_scenario_by_id,
    get_all_scenarios_metadata,
)
from backend.app.models.db_models import (
    TransactionModel,
    RiskAssessmentModel,
    ShapExplanationModel,
    AuditLogModel,
    ModelVersionModel,
    VerificationModel,
)
from backend.app.services.explanation_service import ExplanationService

router = APIRouter()
logger = logging.getLogger("razorguard.api.demo")
explanation_service = ExplanationService(models_dir="models")


@router.get("/demo/scenarios", summary="Get Available Hackathon Demo Scenarios")
def list_demo_scenarios():
    """
    Returns metadata for all reproducible hackathon demo scenarios.
    """
    return {
        "scenarios": get_all_scenarios_metadata(),
        "total_count": len(DEMO_SCENARIOS),
    }


@router.post("/demo/execute/{scenario_id}", summary="Execute Live Demo Scenario with Full Explainability")
def execute_demo_scenario(
    scenario_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Executes a controlled hackathon demo scenario through the real production pipeline:
    Input -> Feature Eng -> ML Prediction -> Deterministic Rules -> SHAP -> LLM -> DB -> Audit Event.
    """
    scenario = get_scenario_by_id(scenario_id)
    if not scenario:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{scenario_id}' not found. Available scenarios: {list(DEMO_SCENARIOS.keys())}",
        )

    request_id = getattr(request.state, "request_id", f"REQ_DEMO_{scenario_id.upper()}")
    tx_dict = scenario["transaction"]
    tx_id = tx_dict["transaction_id"]

    # 1. Run live unified explanation service
    try:
        exp_result = explanation_service.explain_transaction(tx_dict)
    except Exception as e:
        logger.exception("Demo scenario execution failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Demo execution error: {str(e)}") from e

    # 2. Persist transaction & assessment (update or insert cleanly)
    existing_tx = db.query(TransactionModel).filter(TransactionModel.transaction_id == tx_id).first()
    if existing_tx:
        # Clean existing dependent records for fresh execution
        db.query(VerificationModel).filter(VerificationModel.transaction_id == tx_id).delete()
        db.query(ShapExplanationModel).filter(ShapExplanationModel.transaction_id == tx_id).delete()
        db.query(RiskAssessmentModel).filter(RiskAssessmentModel.transaction_id == tx_id).delete()
        db.query(TransactionModel).filter(TransactionModel.transaction_id == tx_id).delete()
        db.flush()

    # Ensure model version exists
    model_version = db.query(ModelVersionModel).filter(ModelVersionModel.version == settings.MODEL_VERSION).first()
    if not model_version:
        db.add(ModelVersionModel(
            version=settings.MODEL_VERSION,
            model_type="Hybrid ML + deterministic rules",
            is_active=True,
        ))

    tx_record = TransactionModel(
        transaction_id=tx_id,
        customer_id=tx_dict["customer_id"],
        merchant_id=tx_dict["merchant_id"],
        amount=tx_dict["amount"],
        currency=tx_dict.get("currency", "USD"),
        merchant_category=tx_dict.get("merchant_category", "retail"),
        payment_method=tx_dict.get("payment_method", "credit_card"),
        country=tx_dict.get("country", "US"),
        raw_payload_json=json.dumps(tx_dict),
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
        fallback_active=exp_result["fallback_active"],
    )
    db.add(assessment_record)

    shap_record = ShapExplanationModel(
        transaction_id=tx_id,
        risk_factors_json=json.dumps(exp_result["risk_factors"]),
        protective_factors_json=json.dumps(exp_result["protective_factors"]),
        llm_explanation_json=json.dumps(exp_result["llm_explanation"]),
    )
    db.add(shap_record)

    # 3. Create Audit Log Entry
    audit_record = AuditLogModel(
        request_id=request_id,
        event_type="DEMO_SCENARIO_EXECUTED",
        transaction_id=tx_id,
        model_version=settings.MODEL_VERSION,
        fraud_probability=exp_result["fraud_probability"],
        risk_score=exp_result["final_risk_score"],
        decision=exp_result["decision"],
        action_taken=f"DEMO_{exp_result['decision']}",
        action=f"DEMO_{exp_result['decision']}",
        actor="DEMO_CONTROLLER",
        performed_by="DEMO_CONTROLLER",
        reason=f"Executed demo {scenario['title']} ({scenario['badge']})",
        top_risk_factors=json.dumps([f["feature_name_human"] for f in exp_result["risk_factors"][:3]]),
        details_json=json.dumps({
            "scenario_id": scenario_id,
            "scenario_title": scenario["title"],
            "demonstration_focus": scenario["demonstration_focus"],
            "triggered_rules": exp_result["triggered_rules"],
        }),
    )
    db.add(audit_record)
    db.commit()
    db.refresh(audit_record)

    return {
        "status": "success",
        "scenario": {
            "id": scenario["id"],
            "title": scenario["title"],
            "badge": scenario["badge"],
            "expected_risk_level": scenario["expected_risk_level"],
            "expected_decision": scenario["expected_decision"],
            "headline": scenario["headline"],
            "demonstration_focus": scenario["demonstration_focus"],
        },
        "input": tx_dict,
        "assessment": {
            "fraud_probability": exp_result["fraud_probability"],
            "ml_risk_score": exp_result["ml_risk_score"],
            "final_risk_score": exp_result["final_risk_score"],
            "anomaly_score": exp_result["anomaly_score"],
            "risk_level": exp_result["risk_level"],
            "decision": exp_result["decision"],
            "triggered_rules": exp_result["triggered_rules"],
            "fallback_active": exp_result["fallback_active"],
            "evaluation_timestamp": exp_result["evaluation_timestamp"],
        },
        "shap_factors": {
            "risk_factors": exp_result["risk_factors"],
            "protective_factors": exp_result["protective_factors"],
        },
        "llm_explanation": exp_result["llm_explanation"],
        "audit_event": {
            "id": audit_record.id,
            "request_id": audit_record.request_id,
            "event_type": audit_record.event_type,
            "action": audit_record.action,
            "actor": audit_record.actor,
            "reason": audit_record.reason,
            "decision": audit_record.decision,
            "risk_score": audit_record.risk_score,
            "timestamp": audit_record.timestamp.isoformat() if audit_record.timestamp else datetime.now(timezone.utc).isoformat(),
        },
    }


@router.post("/demo/reset", summary="Reset Demo State to Clean Baseline")
def reset_demo_state(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Resets demo transactions, related explanations, and registers a demo reset audit log.
    Allows presenters to restart the demonstration cleanly at any time.
    """
    request_id = getattr(request.state, "request_id", "REQ_DEMO_RESET")

    demo_tx_ids = [s["transaction"]["transaction_id"] for s in DEMO_SCENARIOS.values()]

    # Clean up demo specific records
    for tx_id in demo_tx_ids:
        db.query(VerificationModel).filter(VerificationModel.transaction_id == tx_id).delete()
        db.query(ShapExplanationModel).filter(ShapExplanationModel.transaction_id == tx_id).delete()
        db.query(RiskAssessmentModel).filter(RiskAssessmentModel.transaction_id == tx_id).delete()
        db.query(TransactionModel).filter(TransactionModel.transaction_id == tx_id).delete()

    # Log reset audit event
    reset_audit = AuditLogModel(
        request_id=request_id,
        event_type="DEMO_STATE_RESET",
        transaction_id="SYSTEM_RESET",
        model_version=settings.MODEL_VERSION,
        fraud_probability=0.0,
        risk_score=0,
        decision="RESET",
        action_taken="DEMO_RESET",
        action="DEMO_RESET",
        actor="DEMO_CONTROLLER",
        performed_by="DEMO_CONTROLLER",
        reason="Demonstration environment reset to pristine baseline state",
        top_risk_factors="[]",
        details_json=json.dumps({"purged_transactions": demo_tx_ids}),
    )
    db.add(reset_audit)
    db.commit()

    return {
        "status": "success",
        "message": "Demo state successfully reset to clean baseline.",
        "purged_demo_transactions": demo_tx_ids,
        "reset_timestamp": datetime.now(timezone.utc).isoformat(),
    }

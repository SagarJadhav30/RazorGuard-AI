"""Integration and database tests for Phase 9 APIs and Phase 10 persistence."""

import json
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import inspect

from backend.app.core.config import settings
from backend.app.database.session import SessionLocal, engine
from backend.app.main import app
from backend.app.models.db_models import AuditLogModel, ModelVersionModel
from ml.utils.data_generator import get_preset_scenarios


client = TestClient(app)


def make_payload(scenario="standard_grocery_inperson", transaction_id="TXN_PHASE9_10"):
    payload = get_preset_scenarios()[scenario]["data"].copy()
    payload["transaction_id"] = transaction_id
    return payload


def test_phase9_all_required_routes_are_registered():
    routes = set(app.openapi()["paths"])
    assert "/api/v1/risk/predict" in routes
    assert "/api/v1/transactions" in routes
    assert "/api/v1/transactions/{transaction_id}" in routes
    assert "/api/v1/risk/summary" in routes
    assert "/api/v1/model/metrics" in routes
    assert "/api/v1/audit" in routes
    assert "/api/v1/verification/{transaction_id}" in routes
    assert "/api/v1/health" in routes


def test_phase9_prediction_returns_request_id_and_persists_decision():
    response = client.post(
        "/api/v1/risk/predict",
        json=make_payload(transaction_id="TXN_PHASE9_PREDICT"),
        headers={"X-Request-ID": "REQ_PHASE9_PREDICT"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TXN_PHASE9_PREDICT"
    assert body["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert response.headers["X-Request-ID"] == "REQ_PHASE9_PREDICT"

    with SessionLocal() as db:
        audit = db.query(AuditLogModel).filter(
            AuditLogModel.transaction_id == "TXN_PHASE9_PREDICT"
        ).one()
        assert audit.request_id == "REQ_PHASE9_PREDICT"
        assert audit.decision == body["decision"]
        assert audit.fraud_probability == body["fraud_probability"]
        assert audit.model_version == settings.MODEL_VERSION
        assert audit.action == "ASSESS"
        assert audit.actor == "HYBRID_RISK_ENGINE"
        assert audit.reason
        assert json.loads(audit.top_risk_factors) == body["risk_factors"]


def test_phase9_validation_rejects_invalid_and_missing_fields():
    invalid = make_payload(transaction_id="TXN_PHASE9_INVALID")
    invalid["amount"] = 0
    response = client.post("/api/v1/risk/predict", json=invalid)
    assert response.status_code == 422
    assert response.json()["error"] == "Validation Error"
    assert "request_id" in response.json()

    missing = {"amount": 10, "merchant_category": "retail"}
    response = client.post("/api/v1/risk/predict", json=missing)
    assert response.status_code == 422


def test_phase9_transaction_summary_metrics_and_audit_endpoints():
    assert client.get("/api/v1/transactions").status_code == 200
    assert client.get("/api/v1/risk/summary").status_code == 200
    assert client.get("/api/v1/model/metrics").status_code == 200
    audit_response = client.get("/api/v1/audit")
    assert audit_response.status_code == 200
    assert "items" in audit_response.json()


def test_phase9_verification_creates_append_only_audit_event():
    transaction_id = f"TXN_PHASE9_VERIFY_{uuid4().hex[:8]}"
    prediction = client.post(
        "/api/v1/risk/predict", json=make_payload(transaction_id=transaction_id)
    )
    assert prediction.status_code == 200

    response = client.post(
        f"/api/v1/verification/{transaction_id}",
        json={
            "analyst_action": "CONFIRM_LEGIT",
            "analyst_notes": "Synthetic test verification",
            "analyst_id": "TEST_ANALYST",
        },
    )
    assert response.status_code == 200

    with SessionLocal() as db:
        events = db.query(AuditLogModel).filter(
            AuditLogModel.transaction_id == transaction_id
        ).order_by(AuditLogModel.id).all()
        assert len(events) == 2
        assert events[0].action == "ASSESS"
        assert events[1].action == "VERIFY"
        assert events[1].actor == "TEST_ANALYST"
        assert events[1].reason == "Synthetic test verification"
        assert events[0].id < events[1].id


def test_phase10_required_tables_and_indexes_exist():
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    assert {
        "transactions",
        "risk_predictions",
        "risk_factors",
        "verification_actions",
        "audit_logs",
        "model_versions",
    }.issubset(table_names)

    audit_indexes = {index["name"] for index in inspector.get_indexes("audit_logs")}
    assert "ix_audit_logs_transaction_timestamp" in audit_indexes
    assert "ix_audit_logs_decision_timestamp" in audit_indexes


def test_phase10_model_version_is_recorded():
    with SessionLocal() as db:
        model_version = db.query(ModelVersionModel).filter(
            ModelVersionModel.version == settings.MODEL_VERSION
        ).first()
        assert model_version is not None
        assert model_version.is_active is True
        assert model_version.model_type


def test_phase10_sensitive_payment_fields_are_not_stored():
    payload = make_payload(transaction_id="TXN_PHASE10_NO_CARD_DATA")
    payload["card_number"] = "4111111111111111"
    payload["cvv"] = "123"

    response = client.post("/api/v1/risk/predict", json=payload)
    assert response.status_code == 200

    detail = client.get("/api/v1/transactions/TXN_PHASE10_NO_CARD_DATA")
    assert detail.status_code == 200
    raw_payload = detail.json()["raw_payload"]
    assert "card_number" not in raw_payload
    assert "cvv" not in raw_payload

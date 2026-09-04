"""
RazorGuard AI - Phase 9 API Integration Tests
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from backend.app.main import app
from backend.app.database.session import get_db
from backend.app.api.v1.endpoints.risk import explanation_service
from ml.utils.data_generator import get_preset_scenarios

client = TestClient(app)


def test_health_check_endpoints():
    """Verify GET /health and GET /api/v1/health return 200 OK."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

    resp_v1 = client.get("/api/v1/health")
    assert resp_v1.status_code == 200
    assert resp_v1.json()["database_status"] == "connected"


def test_predict_valid_transaction():
    """Verify POST /api/v1/risk/predict assesses valid transaction and returns full JSON payload."""
    presets = get_preset_scenarios()
    payload = presets["stolen_card_attack"]["data"].copy()

    resp = client.post("/api/v1/risk/predict", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "transaction_id" in data
    assert "fraud_probability" in data
    assert "final_risk_score" in data
    assert "risk_level" in data
    assert "decision" in data
    assert "risk_factors" in data
    assert "llm_explanation" in data
    assert "X-Request-ID" in resp.headers


def test_predict_invalid_transaction_negative_amount():
    """Verify POST /api/v1/risk/predict rejects negative amount with 422 Unprocessable Entity."""
    presets = get_preset_scenarios()
    invalid_payload = presets["standard_grocery_inperson"]["data"].copy()
    invalid_payload["amount"] = -50.0  # Invalid negative amount

    resp = client.post("/api/v1/risk/predict", json=invalid_payload)
    assert resp.status_code == 422


def test_predict_missing_required_fields():
    """Verify POST /api/v1/risk/predict rejects payload with missing customer_id."""
    invalid_payload = {
        "amount": 100.0,
        "merchant_category": "retail"
    }
    resp = client.post("/api/v1/risk/predict", json=invalid_payload)
    assert resp.status_code == 422
    assert resp.json()["error"] == "Validation Error"
    assert "request_id" in resp.json()


def test_model_unavailable_uses_conservative_fallback(monkeypatch):
    """A missing model must produce a deterministic, marked assessment."""
    def fail_load():
        raise FileNotFoundError("model artifact missing")

    monkeypatch.setattr(explanation_service.risk_engine.predictor, "is_loaded", False)
    monkeypatch.setattr(explanation_service.risk_engine.predictor, "load", fail_load)
    payload = get_preset_scenarios()["standard_grocery_inperson"]["data"].copy()
    payload["transaction_id"] = "TXN_MODEL_UNAVAILABLE"

    resp = client.post("/api/v1/risk/predict", json=payload)

    assert resp.status_code == 200
    assert resp.json()["fallback_active"] is True


def test_llm_unavailable_uses_offline_explanation(monkeypatch):
    """An optional GenAI outage must not prevent a risk assessment."""
    def fail_llm(*args, **kwargs):
        raise RuntimeError("LLM provider unavailable")

    monkeypatch.setattr(explanation_service.llm_analyst, "generate_analyst_explanation", fail_llm)
    payload = get_preset_scenarios()["standard_grocery_inperson"]["data"].copy()
    payload["transaction_id"] = "TXN_LLM_UNAVAILABLE"

    resp = client.post("/api/v1/risk/predict", json=payload)

    assert resp.status_code == 200
    assert resp.json()["llm_explanation"]["provider_used"] == "offline_deterministic_template"


def test_database_unavailable_returns_structured_503(monkeypatch):
    """Database dependency failures must be reported without leaking internals."""
    def fail_db():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))
        yield

    app.dependency_overrides[get_db] = fail_db
    try:
        resp = client.get("/api/v1/risk/summary", headers={"X-Request-ID": "REQ_DB_TEST"})
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 503
    assert resp.json()["error"] == "Database Unavailable"
    assert resp.json()["request_id"] == "REQ_DB_TEST"


def test_list_and_detail_transactions():
    """Verify listing and retrieving single transaction details."""
    presets = get_preset_scenarios()
    payload = presets["standard_grocery_inperson"]["data"].copy()
    tx_id = "TXN_TEST_9999"
    payload["transaction_id"] = tx_id

    # 1. Post transaction
    client.post("/api/v1/risk/predict", json=payload)

    # 2. List transactions
    resp_list = client.get("/api/v1/transactions")
    assert resp_list.status_code == 200
    list_data = resp_list.json()
    assert list_data["total_count"] >= 1

    # 3. Get single transaction detail
    resp_detail = client.get(f"/api/v1/transactions/{tx_id}")
    assert resp_detail.status_code == 200
    detail_data = resp_detail.json()
    assert detail_data["transaction_id"] == tx_id
    assert "assessment" in detail_data


def test_risk_summary_analytics():
    """Verify GET /api/v1/risk/summary returns aggregate risk stats."""
    resp = client.get("/api/v1/risk/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_transactions_assessed" in data
    assert "approved_count" in data
    assert "total_volume_processed_usd" in data


def test_model_metrics():
    """Verify GET /api/v1/model/metrics returns held-out evaluation metrics."""
    resp = client.get("/api/v1/model/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "roc_auc" in data["metrics"]


def test_audit_logs():
    """Verify GET /api/v1/audit returns immutable audit log list."""
    resp = client.get("/api/v1/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 1


def test_analyst_verification_override():
    """Verify POST /api/v1/verification/{id} submits analyst override."""
    presets = get_preset_scenarios()
    payload = presets["stolen_card_attack"]["data"].copy()
    tx_id = "TXN_OVERRIDE_123"
    payload["transaction_id"] = tx_id

    # 1. Create prediction
    client.post("/api/v1/risk/predict", json=payload)

    # 2. Submit analyst override to CONFIRM_LEGIT
    ver_payload = {
        "analyst_action": "CONFIRM_LEGIT",
        "analyst_notes": "Verified customer identity via 2FA call",
        "analyst_id": "ANALYST_SARAH_02"
    }

    resp_ver = client.post(f"/api/v1/verification/{tx_id}", json=ver_payload)
    assert resp_ver.status_code == 200
    ver_data = resp_ver.json()
    assert ver_data["updated_decision"] == "APPROVE"

    # 3. Check detailed view reflects update
    resp_detail = client.get(f"/api/v1/transactions/{tx_id}")
    assert resp_detail.json()["assessment"]["decision"] == "APPROVE"

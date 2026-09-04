"""
RazorGuard AI - Backend Health Check Tests
"""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify GET / returns welcome JSON."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "RazorGuard AI" in data["message"]


def test_top_level_health_endpoint():
    """Verify GET /health returns 200 healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project_name"] == "RazorGuard AI"


def test_api_v1_health_endpoint():
    """Verify GET /api/v1/health returns detailed health response with database status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database_status" in data
    assert data["database_status"] == "connected"
    assert "timestamp" in data

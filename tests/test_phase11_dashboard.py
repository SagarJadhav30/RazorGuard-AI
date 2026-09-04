"""Contract tests for the Phase 11 merchant dashboard frontend."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
FRONTEND = ROOT / "frontend"
APP_SOURCE = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
STYLES = (FRONTEND / "src" / "index.css").read_text(encoding="utf-8")
PACKAGE = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))


def test_phase11_frontend_stack_is_configured():
    dependencies = PACKAGE["dependencies"]
    assert "react" in dependencies
    assert "recharts" in dependencies
    assert "axios" in dependencies
    assert "lucide-react" in dependencies
    assert PACKAGE["scripts"]["build"] == "vite build"


def test_phase11_dashboard_calls_live_backend_endpoints():
    for endpoint in (
        "/risk/summary",
        "/transactions?limit=500",
        "/model/metrics",
        "/audit?limit=100",
        "/health",
    ):
        assert f"${{API}}{endpoint}" in APP_SOURCE


def test_phase11_requested_views_and_navigation_exist():
    for label in ("Dashboard", "Transactions", "Risk Analytics", "Model Performance", "Audit Trail"):
        assert f"'{label}'" in APP_SOURCE or f'"{label}"' in APP_SOURCE
    assert "Transaction Investigation" in APP_SOURCE
    assert "setSelectedId" in APP_SOURCE
    assert "setPage('investigation')" in APP_SOURCE


def test_phase11_does_not_render_fake_data_when_api_is_empty():
    assert "No backend data available" in APP_SOURCE
    assert "No audit records available" in APP_SOURCE
    assert "No model metrics available" in APP_SOURCE
    assert "Not supplied by API" in APP_SOURCE
    assert "data.transactions" in APP_SOURCE


def test_phase11_dashboard_contains_requested_metrics_and_charts():
    for label in (
        "Total transactions",
        "High risk transactions",
        "Fraud detected",
        "Review queue",
        "Estimated loss prevented",
        "False positive rate",
        "Risk distribution",
        "Fraud vs legitimate",
        "Transaction volume",
        "Fraud trend",
        "Decision distribution",
    ):
        assert label in APP_SOURCE
    assert "ResponsiveContainer" in APP_SOURCE
    assert "PieChart" in APP_SOURCE
    assert "AreaChart" in APP_SOURCE
    assert "LineChart" in APP_SOURCE


def test_phase11_styles_are_responsive_and_fintech_dark():
    assert ".sidebar" in STYLES
    assert ".metric-card" in STYLES
    assert ".risk-badge" in STYLES
    assert "@media(max-width:720px)" in STYLES
    assert "#08111c" in STYLES

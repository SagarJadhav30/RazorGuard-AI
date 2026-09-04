"""
RazorGuard AI - Phase 3 Exploratory Data Analysis Unit Tests
"""

import os
import json
import pytest

from ml.evaluation.eda_analyzer import run_comprehensive_eda


def test_eda_analysis_execution_and_metrics():
    """Verify EDA engine processes dataset and outputs exact statistical metrics."""
    summary = run_comprehensive_eda()

    assert summary["dataset_dimensions"]["records"] == 100000
    assert summary["dataset_dimensions"]["features"] == 27
    assert summary["quality"]["missing_values"] == 0
    assert summary["quality"]["duplicate_records"] == 0
    assert summary["class_imbalance"]["fraud"] == 4000
    assert summary["class_imbalance"]["fraud_rate_pct"] == 4.0


def test_eda_charts_and_reports_generated():
    """Verify charts, markdown report, and notebook artifacts are created."""
    charts_dir = os.path.join("docs", "eda_charts")
    expected_charts = [
        "class_imbalance.png",
        "amount_distribution.png",
        "account_age_vs_fraud.png",
        "merchant_category_fraud.png",
        "hourly_fraud_patterns.png",
        "correlation_heatmap.png"
    ]

    for chart in expected_charts:
        chart_path = os.path.join(charts_dir, chart)
        assert os.path.exists(chart_path), f"Missing chart: {chart_path}"

    assert os.path.exists(os.path.join("docs", "eda_report.md"))
    assert os.path.exists(os.path.join("notebooks", "01_exploratory_data_analysis.ipynb"))

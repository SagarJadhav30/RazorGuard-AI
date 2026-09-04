"""
RazorGuard AI - API v1 Router Aggregator
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, risk, transactions, audit, demo

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(risk.router, tags=["Risk Engine & Prediction"])
api_router.include_router(transactions.router, tags=["Transactions & Verification"])
api_router.include_router(audit.router, tags=["Audit Trail & Logging"])
api_router.include_router(demo.router, tags=["Hackathon Demo Scenarios"])


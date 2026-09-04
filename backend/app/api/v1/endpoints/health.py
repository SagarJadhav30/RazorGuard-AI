"""
RazorGuard AI - Health Endpoint
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.schemas.health import HealthResponse
from backend.app.core.config import settings
from backend.app.database.session import get_db

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Check API & Database Health")
def check_health(db: Session = Depends(get_db)):
    """
    Returns system operational health, version, environment, and database status.
    """
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"disconnected ({str(e)})"

    return HealthResponse(
        status="healthy",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
        database_status=db_status
    )

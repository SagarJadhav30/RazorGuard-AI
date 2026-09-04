"""
RazorGuard AI - Health Check Schema
"""

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    project_name: str = Field(..., json_schema_extra={"example": "RazorGuard AI"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    environment: str = Field(..., json_schema_extra={"example": "development"})
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    database_status: str = Field(..., json_schema_extra={"example": "connected"})

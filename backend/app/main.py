"""
RazorGuard AI - FastAPI Main Application Entrypoint
"""

import uuid
import logging
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import settings
from backend.app.database.session import engine, Base, upgrade_legacy_sqlite_schema
from backend.app.api.v1.router import api_router

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("razorguard.api")

# Automatically create database tables on startup
Base.metadata.create_all(bind=engine)
upgrade_legacy_sqlite_schema()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Explainable AI Risk Manager for Payment Fraud Detection (Defense-Only)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 1. Request ID Middleware
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", f"REQ_{uuid.uuid4().hex[:12]}")
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

# 2. Global Unhandled Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    logger.error(f"Unhandled Exception [Request-ID: {req_id}]: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during processing.",
            "request_id": req_id
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "The request contains invalid or missing fields.",
            "details": exc.errors(),
            "request_id": req_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    message = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Request Error", "message": message, "request_id": req_id},
        headers=exc.headers,
    )


@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(request: Request, exc: SQLAlchemyError):
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    logger.error("Database operation failed [Request-ID: %s]", req_id, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Database Unavailable",
            "message": "The request could not be completed because the database is unavailable.",
            "request_id": req_id,
        },
    )

# 3. Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Health"])
def root_health():
    return {
        "status": "healthy",
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to RazorGuard AI Risk Manager API",
        "docs": "/docs",
        "health": "/health",
        "api_v1": f"{settings.API_V1_STR}/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

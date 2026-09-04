"""
RazorGuard AI - Database Session & Engine Setup
Supports PostgreSQL and SQLite (fallback) with SQLAlchemy ORM.
"""

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from backend.app.core.config import settings

database_url = settings.DATABASE_URL

# Configure engine kwargs depending on SQLite vs PostgreSQL
engine_kwargs = {}
if database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(database_url, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base ORM model class"""
    pass


def get_db():
    """Dependency injection generator for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def upgrade_legacy_sqlite_schema():
    """Add Phase 10 audit columns to an existing local SQLite database."""
    if not database_url.startswith("sqlite"):
        return

    columns = {
        column["name"] for column in inspect(engine).get_columns("audit_logs")
    }
    upgrades = {
        "model_version": "VARCHAR(64) NOT NULL DEFAULT 'legacy'",
        "fraud_probability": "FLOAT",
        "action": "VARCHAR(50) NOT NULL DEFAULT 'ASSESS'",
        "actor": "VARCHAR(64) NOT NULL DEFAULT 'SYSTEM_RISK_ENGINE'",
        "reason": "TEXT NOT NULL DEFAULT 'Legacy audit record'",
        "top_risk_factors": "TEXT NOT NULL DEFAULT '[]'",
    }
    missing = [(name, definition) for name, definition in upgrades.items() if name not in columns]
    with engine.begin() as connection:
        for name, definition in missing:
            connection.exec_driver_sql(
                f"ALTER TABLE audit_logs ADD COLUMN {name} {definition}"
            )
        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_transaction_timestamp "
            "ON audit_logs (transaction_id, timestamp)"
        )
        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_decision_timestamp "
            "ON audit_logs (decision, timestamp)"
        )

"""
RazorGuard AI - Core Configuration Settings
"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "RazorGuard AI"
    VERSION: str = "1.0.0"
    MODEL_VERSION: str = "risk-model-v1"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173"
    ]

    # Database Configuration (Defaults to SQLite for instant local dev, PostgreSQL ready)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./razorguard.db")

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "offline_template")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

"""Application configuration from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    vectorsight_env: str = "development"
    vectorsight_log_level: str = "debug"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Model routing
    model_cheap: str = "claude-haiku-4-5-20251001"
    model_mid: str = "claude-sonnet-4-5-20250929"
    model_frontier: str = "claude-sonnet-4-5-20250929"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

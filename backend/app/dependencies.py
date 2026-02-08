"""FastAPI dependency injection."""

from __future__ import annotations

from app.config import settings


def get_settings():
    return settings

"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.engine.registry import get_registry
from app.models.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.1.0",
        transforms_registered=get_registry().count,
    )

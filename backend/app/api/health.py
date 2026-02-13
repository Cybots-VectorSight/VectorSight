"""Health check + meta endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.responses import HealthResponse

router = APIRouter()

# The breakdown pipeline has 14 steps across 3 stages
_BREAKDOWN_STEPS = 14


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.1.0",
        transforms_registered=_BREAKDOWN_STEPS,
    )


@router.get("/prompts")
async def prompts() -> dict[str, str]:
    from app.llm.prompts import get_all_templates

    return get_all_templates()

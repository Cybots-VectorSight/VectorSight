"""POST /api/analyze — full pipeline analysis."""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.engine.pipeline import create_pipeline
from app.models.requests import AnalyzeRequest
from app.models.responses import AnalyzeResponse
from app.svg.parser import parse_svg

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    start = time.perf_counter()

    # Parse SVG into PipelineContext
    ctx = parse_svg(req.svg)

    # Run full pipeline
    pipeline = create_pipeline()
    ctx = pipeline.run(ctx)

    elapsed = (time.perf_counter() - start) * 1000

    # Build enrichment output from context
    from app.llm.enrichment_formatter import context_to_enrichment

    enrichment = context_to_enrichment(ctx)

    return AnalyzeResponse(
        enrichment=enrichment,
        processing_time_ms=round(elapsed, 1),
        transforms_completed=len(ctx.completed_transforms),
        transforms_failed=len(ctx.errors),
        errors=ctx.errors,
    )

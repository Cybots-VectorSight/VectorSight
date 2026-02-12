"""POST /api/analyze — full pipeline analysis."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.engine.pipeline import create_pipeline
from app.models.requests import AnalyzeRequest
from app.models.responses import AnalyzeResponse
from app.svg.parser import parse_svg

router = APIRouter()


_SENTINEL = object()  # marks end of queue


async def _stream_analyze(svg: str) -> AsyncGenerator[str, None]:
    """Drive pipeline.run_streaming() in a thread, yielding SSE events as they arrive."""
    start = time.perf_counter()

    try:
        ctx = parse_svg(svg)
    except Exception as e:
        data = json.dumps({"type": "error", "message": str(e)})
        yield f"event: error\ndata: {data}\n\n"
        return

    pipeline = create_pipeline()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_pipeline() -> None:
        """Sync pipeline in thread — pushes progress dicts onto the async queue."""
        for progress in pipeline.run_streaming(ctx):
            loop.call_soon_threadsafe(queue.put_nowait, progress)
        loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    # Start pipeline in a thread so the event loop stays free to flush SSE
    asyncio.get_running_loop().run_in_executor(None, _run_pipeline)

    # Drain the queue, yielding SSE events as they arrive
    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        data = json.dumps(item)
        yield f"event: progress\ndata: {data}\n\n"

    # Build enrichment after pipeline completes
    elapsed = (time.perf_counter() - start) * 1000

    from app.llm.enrichment_formatter import context_to_enrichment

    enrichment = context_to_enrichment(ctx)

    response = AnalyzeResponse(
        enrichment=enrichment,
        processing_time_ms=round(elapsed, 1),
        transforms_completed=len(ctx.completed_transforms),
        transforms_failed=len(ctx.errors),
        errors=ctx.errors,
    )

    # Estimate tokens: ~1.3 tokens per word in enrichment text
    word_count = len((enrichment.enrichment_text or "").split())
    estimated_tokens = round(word_count * 1.3)

    result_data = response.model_dump()
    result_data["estimated_tokens"] = estimated_tokens
    yield f"event: result\ndata: {json.dumps(result_data)}\n\n"

    yield f"event: done\ndata: {json.dumps({'type': 'done'})}\n\n"


@router.post("/analyze/stream")
async def analyze_stream(req: AnalyzeRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_analyze(req.svg),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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

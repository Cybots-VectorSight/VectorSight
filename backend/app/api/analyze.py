"""POST /api/analyze -- full pipeline analysis."""

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

router = APIRouter()


_SENTINEL = object()  # marks end of queue


async def _stream_analyze(svg: str) -> AsyncGenerator[str, None]:
    """Drive pipeline.run_streaming() in a thread, yielding SSE events as they arrive."""
    start = time.perf_counter()

    pipeline = create_pipeline()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_pipeline() -> None:
        """Sync pipeline in thread -- pushes progress dicts onto the async queue."""
        try:
            for progress in pipeline.run_streaming(svg):
                loop.call_soon_threadsafe(queue.put_nowait, progress)
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            loop.call_soon_threadsafe(queue.put_nowait, error_event)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    # Start pipeline in a thread so the event loop stays free to flush SSE
    asyncio.get_running_loop().run_in_executor(None, _run_pipeline)

    # Drain the queue, yielding SSE events as they arrive
    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, dict) and item.get("type") == "step_visual":
            yield f"event: step_visual\ndata: {json.dumps(item)}\n\n"
        else:
            data = json.dumps(item)
            yield f"event: progress\ndata: {data}\n\n"

    # Build response after pipeline completes
    elapsed = (time.perf_counter() - start) * 1000
    result = pipeline.result

    if result is None:
        data = json.dumps({"type": "error", "message": "Pipeline produced no result"})
        yield f"event: error\ndata: {data}\n\n"
        return

    response = AnalyzeResponse(
        enrichment=result.enrichment_output,
        processing_time_ms=round(elapsed, 1),
        transforms_completed=len(result.completed_steps),
        transforms_failed=len(result.errors),
        errors=result.errors,
    )

    # Estimate tokens: ~1.3 tokens per word in enrichment text
    word_count = len((result.enrichment_text or "").split())
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

    # Run full pipeline
    pipeline = create_pipeline()
    result = pipeline.run(req.svg)

    elapsed = (time.perf_counter() - start) * 1000

    return AnalyzeResponse(
        enrichment=result.enrichment_output,
        processing_time_ms=round(elapsed, 1),
        transforms_completed=len(result.completed_steps),
        transforms_failed=len(result.errors),
        errors=result.errors,
    )

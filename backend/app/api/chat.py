"""POST /api/chat — SVG analysis + LLM Q&A (standard + streaming)."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.models.requests import ChatRequest
from app.models.responses import ChatResponse

router = APIRouter()


def _run_pipeline(svg: str):
    """Run the full pipeline and return (ctx, enrichment_text)."""
    from app.engine.pipeline import create_pipeline
    from app.llm.enrichment_formatter import context_to_enrichment_text
    from app.svg.parser import parse_svg

    ctx = parse_svg(svg)
    pipeline = create_pipeline()
    ctx = pipeline.run(ctx)
    return ctx, context_to_enrichment_text(ctx)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    from app.llm.client import get_chat_response

    if req.enrichment:
        enrichment_text = req.enrichment
        ctx = None
    else:
        ctx, enrichment_text = _run_pipeline(req.svg)

    answer = await get_chat_response(
        svg=req.svg,
        enrichment=enrichment_text,
        question=req.question,
        history=req.history,
        task="chat",
    )

    # Auto-record session for learning (skip if no pipeline context)
    if ctx is not None:
        from app.learning.memory import record_from_context

        record_from_context(ctx, svg=req.svg, question=req.question, answer=answer)

        from app.learning.self_reflect import reflect_background_chat

        background_tasks.add_task(
            reflect_background_chat,
            req.svg, enrichment_text, req.question, answer, ctx,
        )

    return ChatResponse(answer=answer, enrichment_used=True)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    from app.llm.stream import stream_chat_response

    if req.enrichment:
        enrichment_text = req.enrichment
        ctx = None
    else:
        ctx, enrichment_text = _run_pipeline(req.svg)

    # Auto-record session for learning (skip if no pipeline context)
    if ctx is not None:
        from app.learning.memory import record_from_context

        record_from_context(ctx, svg=req.svg, question=req.question)

        from app.learning.self_reflect import reflect_background_chat

        background_tasks.add_task(
            reflect_background_chat,
            req.svg, enrichment_text, req.question, "(streaming — answer not captured)", ctx,
        )

    return StreamingResponse(
        stream_chat_response(
            svg=req.svg,
            enrichment=enrichment_text,
            question=req.question,
            history=req.history,
            task="chat",
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

"""POST /api/chat — SVG analysis + LLM Q&A (standard + streaming)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.requests import ChatRequest
from app.models.responses import ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    from app.engine.pipeline import create_pipeline
    from app.llm.client import get_chat_response
    from app.llm.enrichment_formatter import context_to_enrichment_text
    from app.svg.parser import parse_svg

    ctx = parse_svg(req.svg)
    pipeline = create_pipeline()
    ctx = pipeline.run(ctx)

    enrichment_text = context_to_enrichment_text(ctx)
    answer = await get_chat_response(
        svg=req.svg,
        enrichment=enrichment_text,
        question=req.question,
        history=req.history,
        task="chat",
    )

    # Auto-record session for learning
    from app.learning.memory import record_from_context

    record_from_context(ctx, svg=req.svg, question=req.question, answer=answer)

    return ChatResponse(answer=answer, enrichment_used=True)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    from app.engine.pipeline import create_pipeline
    from app.llm.enrichment_formatter import context_to_enrichment_text
    from app.llm.stream import stream_chat_response
    from app.svg.parser import parse_svg

    ctx = parse_svg(req.svg)
    pipeline = create_pipeline()
    ctx = pipeline.run(ctx)

    enrichment_text = context_to_enrichment_text(ctx)

    # Auto-record session for learning (question only — streaming answer not captured)
    from app.learning.memory import record_from_context

    record_from_context(ctx, svg=req.svg, question=req.question)

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

"""POST /api/chat -- SVG analysis + LLM Q&A (standard + streaming)."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.models.requests import ChatRequest
from app.models.responses import ChatResponse

router = APIRouter()


def _run_pipeline(svg: str):
    """Run the full pipeline and return (result, enrichment_text)."""
    from app.engine.pipeline import create_pipeline

    pipeline = create_pipeline()
    result = pipeline.run(svg)
    return result, result.enrichment_text


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    from app.llm.client import get_chat_response

    if req.enrichment is not None:
        enrichment_text = req.enrichment
        breakdown_result = None
    else:
        breakdown_result, enrichment_text = _run_pipeline(req.svg)

    answer = await get_chat_response(
        svg=req.svg,
        enrichment=enrichment_text,
        question=req.question,
        history=req.history,
        task="chat",
    )

    # Auto-record session for learning (skip if no pipeline result)
    if breakdown_result is not None:
        from app.learning.memory import record_from_breakdown

        record_from_breakdown(
            breakdown_result, svg=req.svg, question=req.question, answer=answer
        )

        from app.learning.self_reflect import reflect_background_chat

        background_tasks.add_task(
            reflect_background_chat,
            req.svg, enrichment_text, req.question, answer,
        )

    return ChatResponse(answer=answer, enrichment_used=True)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    from app.llm.stream import stream_chat_response

    if req.enrichment is not None:
        enrichment_text = req.enrichment
        breakdown_result = None
    else:
        breakdown_result, enrichment_text = _run_pipeline(req.svg)

    # Auto-record session for learning (skip if no pipeline result)
    if breakdown_result is not None:
        from app.learning.memory import record_from_breakdown

        record_from_breakdown(
            breakdown_result, svg=req.svg, question=req.question
        )

        from app.learning.self_reflect import reflect_background_chat

        background_tasks.add_task(
            reflect_background_chat,
            req.svg, enrichment_text, req.question, "(streaming -- answer not captured)",
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

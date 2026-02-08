"""POST /api/modify — SVG modification via spatial intent."""

from __future__ import annotations

import re

from fastapi import APIRouter, BackgroundTasks

from app.models.requests import ModifyRequest
from app.models.responses import ModifyResponse

router = APIRouter()


def _extract_svg(text: str) -> str:
    """Extract SVG from LLM output, stripping markdown fences and surrounding text."""
    # Remove markdown code fences
    stripped = re.sub(r"^```(?:xml|svg|html)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    stripped = stripped.strip()

    # Extract <svg>...</svg> if there's text before/after
    match = re.search(r"(<svg[\s\S]*</svg>)", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return stripped


@router.post("/modify", response_model=ModifyResponse)
async def modify(req: ModifyRequest, background_tasks: BackgroundTasks) -> ModifyResponse:
    from app.engine.pipeline import create_pipeline
    from app.llm.client import get_chat_response
    from app.llm.enrichment_formatter import context_to_enrichment_text
    from app.svg.parser import parse_svg

    ctx = parse_svg(req.svg)
    pipeline = create_pipeline()
    ctx = pipeline.run(ctx)

    enrichment_text = context_to_enrichment_text(ctx)
    result = await get_chat_response(
        svg=req.svg,
        enrichment=enrichment_text,
        question=req.instruction,
        history=req.history,
        task="modify",
    )

    clean_svg = _extract_svg(result)

    # Auto-record session for learning
    from app.learning.memory import record_from_context

    record_from_context(ctx, svg=req.svg, question=req.instruction)

    # Self-reflect: LLM vision sees the rendered SVG and auto-learns
    from app.learning.self_reflect import reflect_background

    background_tasks.add_task(reflect_background, req.svg, enrichment_text, ctx)

    return ModifyResponse(svg=clean_svg, changes=[req.instruction])

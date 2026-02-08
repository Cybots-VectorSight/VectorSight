"""POST /api/modify — SVG modification via spatial intent."""

from __future__ import annotations

import re

from fastapi import APIRouter

from app.models.requests import ModifyRequest
from app.models.responses import ModifyResponse

router = APIRouter()


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```xml ... ```) wrapping SVG output."""
    stripped = re.sub(r"^```(?:xml|svg|html)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


@router.post("/modify", response_model=ModifyResponse)
async def modify(req: ModifyRequest) -> ModifyResponse:
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
        history=[],
        task="modify",
    )

    clean_svg = _strip_markdown_fences(result)
    return ModifyResponse(svg=clean_svg, changes=[req.instruction])

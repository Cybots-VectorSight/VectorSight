"""POST /api/playground/* — interactive SVG playground."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.requests import PlaygroundClickRequest
from app.models.responses import PlaygroundResponse

router = APIRouter(prefix="/playground")


@router.post("/click", response_model=PlaygroundResponse)
async def handle_click(req: PlaygroundClickRequest) -> PlaygroundResponse:
    from app.engine.pipeline import create_pipeline
    from app.llm.client import get_chat_response
    from app.llm.enrichment_formatter import context_to_enrichment_text
    from app.svg.parser import parse_svg

    ctx = parse_svg(req.svg)
    pipeline = create_pipeline()
    ctx = pipeline.run(ctx)

    # Find which element was clicked
    clicked_element = None
    for sp in ctx.subpaths:
        xmin, ymin, xmax, ymax = sp.bbox
        if xmin <= req.x <= xmax and ymin <= req.y <= ymax:
            clicked_element = sp.id
            break

    enrichment_text = context_to_enrichment_text(ctx)
    instruction = f"The user clicked on element {clicked_element or 'empty space'} at ({req.x}, {req.y}). Do something creative with it."

    result = await get_chat_response(
        svg=req.svg,
        enrichment=enrichment_text,
        question=instruction,
        history=[],
        task="playground",
    )

    return PlaygroundResponse(
        svg=result,
        action="creative modification",
        element_clicked=clicked_element,
    )

"""POST /api/playground/* -- interactive SVG playground."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.requests import PlaygroundClickRequest
from app.models.responses import PlaygroundResponse

router = APIRouter(prefix="/playground")


@router.post("/click", response_model=PlaygroundResponse)
async def handle_click(req: PlaygroundClickRequest) -> PlaygroundResponse:
    from app.engine.pipeline import create_pipeline
    from app.llm.client import get_chat_response

    pipeline = create_pipeline()
    breakdown_result = pipeline.run(req.svg)

    enrichment_text = breakdown_result.enrichment_text

    # Find which group the click falls in
    clicked_element = None
    from shapely.geometry import Point

    click_pt = Point(req.x, req.y)
    for gi, g in enumerate(breakdown_result.groups):
        if g.polygon and g.polygon.contains(click_pt):
            clicked_element = f"G{gi}"

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

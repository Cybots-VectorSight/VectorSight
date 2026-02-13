"""POST /api/icon-set/* -- icon set analysis and generation."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.requests import IconSetAnalyzeRequest, IconSetGenerateRequest
from app.models.responses import IconSetGenerateResponse, IconSetRulesResponse

router = APIRouter(prefix="/icon-set")


@router.post("/analyze", response_model=IconSetRulesResponse)
async def analyze_set(req: IconSetAnalyzeRequest) -> IconSetRulesResponse:
    from app.engine.pipeline import create_pipeline

    rules: dict[str, str] = {}
    common_props: dict[str, float] = {}

    all_elements = []
    for svg in req.svgs:
        pipeline = create_pipeline()
        result = pipeline.run(svg)
        all_elements.extend(result.enrichment_output.elements)

    if all_elements:
        # Infer style from shapes
        shape_counts: dict[str, int] = {}
        for elem in all_elements:
            shape_counts[elem.shape_class] = shape_counts.get(elem.shape_class, 0) + 1
        if shape_counts:
            rules["dominant_shape"] = max(shape_counts, key=shape_counts.get)

    return IconSetRulesResponse(rules=rules, common_properties=common_props)


@router.post("/generate", response_model=IconSetGenerateResponse)
async def generate_icon(req: IconSetGenerateRequest) -> IconSetGenerateResponse:
    from app.llm.client import get_chat_response

    result = await get_chat_response(
        svg="\n---\n".join(req.svgs),
        enrichment="",
        question=req.description,
        history=[],
        task="icon_set",
    )

    return IconSetGenerateResponse(svg=result, rules_applied=[])

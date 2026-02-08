"""POST /api/icon-set/* — icon set analysis and generation."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.requests import IconSetAnalyzeRequest, IconSetGenerateRequest
from app.models.responses import IconSetGenerateResponse, IconSetRulesResponse

router = APIRouter(prefix="/icon-set")


@router.post("/analyze", response_model=IconSetRulesResponse)
async def analyze_set(req: IconSetAnalyzeRequest) -> IconSetRulesResponse:
    from app.engine.pipeline import create_pipeline
    from app.svg.parser import parse_svg

    pipeline = create_pipeline()
    all_features: list[dict] = []

    for svg in req.svgs:
        ctx = parse_svg(svg)
        ctx = pipeline.run(ctx)
        for sp in ctx.subpaths:
            all_features.append(sp.features)

    # Extract common rules across the set
    rules: dict[str, str] = {}
    common_props: dict[str, float] = {}

    if all_features:
        rules["style"] = "outline" if any(f.get("is_stroke") for f in all_features) else "filled"

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

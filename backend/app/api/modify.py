"""POST /api/modify — SVG modification via surgical edit operations (with full-rewrite fallback)."""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, BackgroundTasks

from app.models.requests import ModifyRequest
from app.models.responses import ModifyResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _extract_svg(text: str) -> str:
    """Extract SVG from LLM output, stripping markdown fences and surrounding text."""
    stripped = re.sub(r"^```(?:xml|svg|html)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    stripped = stripped.strip()

    match = re.search(r"(<svg[\s\S]*</svg>)", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return stripped


def _parse_edit_plan(text: str):
    """Try to parse LLM output as an EditPlan. Returns (EditPlan, None) or (None, error_msg)."""
    from app.models.edit_ops import EditPlan

    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        plan = EditPlan(**data)
        if plan.operations:
            return plan, None
        return None, "EditPlan has no operations"
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return None, str(e)


@router.post("/modify", response_model=ModifyResponse)
async def modify(req: ModifyRequest, background_tasks: BackgroundTasks) -> ModifyResponse:
    from app.llm.client import get_chat_response

    if req.enrichment:
        enrichment_text = req.enrichment
        ctx = None
    else:
        from app.engine.pipeline import create_pipeline
        from app.llm.enrichment_formatter import context_to_enrichment_text
        from app.svg.parser import parse_svg

        ctx = parse_svg(req.svg)
        pipeline = create_pipeline()
        ctx = pipeline.run(ctx)
        enrichment_text = context_to_enrichment_text(ctx)

    # --- Try surgical edit path first ---
    edit_result = await get_chat_response(
        svg=req.svg,
        enrichment=enrichment_text,
        question=req.instruction,
        history=req.history,
        task="edit",
    )

    plan, parse_error = _parse_edit_plan(edit_result)
    clean_svg: str
    edit_ops_debug: list[dict] | None = None

    if plan is not None:
        from app.svg.edit_applier import apply_edits
        from app.svg.parser import parse_svg

        # apply_edits needs a parsed context for element offsets
        edit_ctx = ctx if ctx is not None else parse_svg(req.svg)
        logger.info("Surgical edit: %d operations", len(plan.operations))
        clean_svg = apply_edits(req.svg, plan.operations, edit_ctx)
        edit_ops_debug = [op.model_dump(exclude_none=True) for op in plan.operations]
    else:
        # --- Fallback: full-rewrite via modify prompt ---
        logger.info("Edit plan parse failed (%s), falling back to full rewrite", parse_error)
        fallback_result = await get_chat_response(
            svg=req.svg,
            enrichment=enrichment_text,
            question=req.instruction,
            history=req.history,
            task="modify",
        )
        clean_svg = _extract_svg(fallback_result)

    # Auto-record session for learning (skip if no pipeline context)
    if ctx is not None:
        from app.learning.memory import record_from_context

        record_from_context(ctx, svg=req.svg, question=req.instruction)

        from app.learning.self_reflect import reflect_background_modify

        background_tasks.add_task(
            reflect_background_modify,
            req.svg, clean_svg, enrichment_text, req.instruction, ctx,
        )

    return ModifyResponse(
        svg=clean_svg,
        changes=[req.instruction],
        edit_ops=edit_ops_debug,
        reasoning=plan.reasoning if plan is not None else None,
    )

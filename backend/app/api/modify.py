"""POST /api/modify -- SVG modification via surgical edit operations (with full-rewrite fallback)."""

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

    # Also try to extract JSON from surrounding text
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        cleaned = json_match.group(0)

    try:
        data = json.loads(cleaned)
        plan = EditPlan(**data)
        if plan.operations:
            return plan, None
        return None, "EditPlan has no operations"
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return None, str(e)


def _build_element_listing(
    svg_text: str,
    breakdown_result=None,
) -> str:
    """Build element listing with group labels so LLM knows what each element IS.

    Maps each E# element to its enrichment group (A, B, C...) by spatial overlap,
    so the LLM sees: 'E5: group C "elongated extension" at bottom-left'.
    """
    from app.svg.parser import parse_svg

    try:
        ctx = parse_svg(svg_text)
    except Exception as e:
        logger.warning("parse_svg failed for element listing: %s", e)
        return ""

    if not ctx.subpaths:
        return ""

    canvas_w = ctx.canvas_width or 300
    canvas_h = ctx.canvas_height or 150

    # Build group mapping: for each E# element, find which enrichment group it belongs to
    group_labels: dict[str, str] = {}  # E# -> "C 'elongated extension'"
    if breakdown_result is not None:
        from app.engine.breakdown.prompt_builder import _label, _infer_feature_role
        from shapely.geometry import Point

        groups = breakdown_result.groups or []
        for sp in ctx.subpaths:
            cx, cy = sp.centroid
            pt = Point(cx, cy)
            best_gi = -1
            best_area = float("inf")
            # Find smallest group whose polygon contains this element's centroid
            for gi, g in enumerate(groups):
                if g.polygon is None or g.polygon.is_empty:
                    continue
                try:
                    if g.polygon.contains(pt) and g.area < best_area:
                        best_gi = gi
                        best_area = g.area
                except Exception:
                    continue
            if best_gi >= 0:
                lbl = _label(best_gi)
                role = _infer_feature_role(groups[best_gi], best_gi, canvas_w, canvas_h)
                group_labels[sp.id] = f'{lbl} "{role}"'

    def _pos(cx: float, cy: float) -> str:
        h = "left" if cx < canvas_w / 3 else ("center" if cx < 2 * canvas_w / 3 else "right")
        v = "top" if cy < canvas_h / 3 else ("middle" if cy < 2 * canvas_h / 3 else "bottom")
        return f"{v}-{h}"

    def _tag_name(tag: str) -> str:
        m = re.match(r"<(\w+)", tag)
        return m.group(1) if m else "?"

    def _fill(tag: str) -> str:
        m = re.search(r'fill="([^"]+)"', tag)
        if m:
            f = m.group(1)
            return f if f.lower() not in ("none", "transparent") else ""
        return ""

    lines = [f"ELEMENTS ({len(ctx.subpaths)} total — use these IDs for edit targets):"]
    for sp in ctx.subpaths:
        cx, cy = sp.centroid
        pos = _pos(cx, cy)
        tag_type = _tag_name(sp.source_tag)
        fill = _fill(sp.source_tag)
        w, h = sp.bbox[2] - sp.bbox[0], sp.bbox[3] - sp.bbox[1]

        # Core description
        desc = f"  {sp.id} [{pos}]: {tag_type} {w:.0f}x{h:.0f}px"
        if fill:
            desc += f" fill={fill}"

        # Group mapping — the key info: what IS this element?
        grp = group_labels.get(sp.id)
        if grp:
            desc += f"  → group {grp}"

        lines.append(desc)

    return "\n".join(lines)


def _validate_svg(svg_text: str) -> tuple[bool, str]:
    """Basic SVG validation. Returns (is_valid, error_message)."""
    if not svg_text or not svg_text.strip():
        return False, "Empty SVG output"

    # Must contain <svg and </svg>
    if not re.search(r"<svg[\s>]", svg_text, re.IGNORECASE):
        return False, "Missing <svg> tag"
    if not re.search(r"</svg\s*>", svg_text, re.IGNORECASE):
        return False, "Missing </svg> closing tag"

    # Check for obviously broken XML (unmatched quotes in attributes)
    # Count quotes — should be even
    in_tag = False
    quote_count = 0
    for ch in svg_text:
        if ch == '<':
            in_tag = True
            quote_count = 0
        elif ch == '>':
            if quote_count % 2 != 0:
                return False, "Unmatched quotes in SVG tag"
            in_tag = False
        elif in_tag and ch == '"':
            quote_count += 1

    return True, ""


@router.post("/modify", response_model=ModifyResponse)
async def modify(req: ModifyRequest, background_tasks: BackgroundTasks) -> ModifyResponse:
    from app.llm.client import get_chat_response

    if req.enrichment is not None:
        enrichment_text = req.enrichment
        breakdown_result = None
    else:
        from app.engine.pipeline import create_pipeline

        pipeline = create_pipeline()
        breakdown_result = pipeline.run(req.svg)
        enrichment_text = breakdown_result.enrichment_text

    # Build element listing for surgical edit path (maps E# IDs → group labels)
    element_listing = _build_element_listing(req.svg, breakdown_result)
    edit_enrichment = enrichment_text
    if element_listing:
        edit_enrichment = element_listing + "\n\n" + enrichment_text

    # --- Try surgical edit path first ---
    edit_result = await get_chat_response(
        svg=req.svg,
        enrichment=edit_enrichment,
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

        try:
            edit_ctx = parse_svg(req.svg)
            logger.info("Surgical edit: %d operations", len(plan.operations))
            clean_svg = apply_edits(req.svg, plan.operations, edit_ctx)
            edit_ops_debug = [op.model_dump(exclude_none=True) for op in plan.operations]

            # Validate the result
            valid, err = _validate_svg(clean_svg)
            if not valid:
                logger.warning("Surgical edit produced invalid SVG (%s), falling back to full rewrite", err)
                plan = None  # fall through to full rewrite
        except Exception as e:
            logger.warning("Surgical edit apply failed (%s), falling back to full rewrite", e)
            plan = None  # fall through to full rewrite

    if plan is None:
        # --- Fallback: full-rewrite via modify prompt ---
        logger.info("Edit plan parse failed (%s), falling back to full rewrite", parse_error or "apply error")
        fallback_result = await get_chat_response(
            svg=req.svg,
            enrichment=enrichment_text,
            question=req.instruction,
            history=req.history,
            task="modify",
        )
        clean_svg = _extract_svg(fallback_result)
        edit_ops_debug = None

        # Validate fallback too — if still broken, return original SVG with error
        valid, err = _validate_svg(clean_svg)
        if not valid:
            logger.error("Full rewrite also produced invalid SVG: %s", err)
            clean_svg = req.svg  # return original unchanged

    # Auto-record session for learning
    if breakdown_result is not None:
        from app.learning.memory import record_from_breakdown

        record_from_breakdown(
            breakdown_result, svg=req.svg, question=req.instruction
        )

        from app.learning.self_reflect import reflect_background_modify

        background_tasks.add_task(
            reflect_background_modify,
            req.svg, clean_svg, enrichment_text, req.instruction,
        )

    return ModifyResponse(
        svg=clean_svg,
        changes=[req.instruction],
        edit_ops=edit_ops_debug,
        reasoning=plan.reasoning if plan is not None else None,
    )

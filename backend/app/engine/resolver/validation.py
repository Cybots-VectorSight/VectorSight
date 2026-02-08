"""Validate generated SVG through the forward pipeline."""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.pipeline import Pipeline, create_pipeline
from app.svg.parser import parse_svg


def validate_generated_svg(svg: str) -> dict:
    """Run the generated SVG through the forward pipeline to validate it.

    Returns a dict with:
    - valid: bool
    - element_count: int
    - issues: list[str]
    - enrichment_text: str (if valid)
    """
    try:
        ctx = parse_svg(svg)
    except Exception as e:
        return {
            "valid": False,
            "element_count": 0,
            "issues": [f"Parse error: {e}"],
            "enrichment_text": "",
        }

    if ctx.num_elements == 0:
        return {
            "valid": False,
            "element_count": 0,
            "issues": ["No elements found in SVG"],
            "enrichment_text": "",
        }

    # Run pipeline
    pipeline = create_pipeline()
    ctx = pipeline.run(ctx)

    # Collect validation issues
    issues: list[str] = []
    for sp in ctx.subpaths:
        sp_issues = sp.features.get("consistency_issues", [])
        for issue in sp_issues:
            issues.append(f"{sp.id}: {issue}")

    # Check for pipeline errors
    for tid, err in ctx.errors.items():
        issues.append(f"Transform {tid}: {err}")

    from app.llm.enrichment_formatter import context_to_enrichment_text
    enrichment_text = context_to_enrichment_text(ctx)

    return {
        "valid": len(issues) == 0,
        "element_count": ctx.num_elements,
        "issues": issues,
        "enrichment_text": enrichment_text,
    }

"""Validate generated SVG through the forward pipeline."""

from __future__ import annotations

from app.engine.pipeline import create_pipeline


def validate_generated_svg(svg: str) -> dict:
    """Run the generated SVG through the forward pipeline to validate it.

    Returns a dict with:
    - valid: bool
    - element_count: int
    - issues: list[str]
    - enrichment_text: str (if valid)
    """
    try:
        pipeline = create_pipeline()
        result = pipeline.run(svg)
    except Exception as e:
        return {
            "valid": False,
            "element_count": 0,
            "issues": [f"Pipeline error: {e}"],
            "enrichment_text": "",
        }

    if not result.groups:
        return {
            "valid": False,
            "element_count": 0,
            "issues": ["No elements found in SVG"],
            "enrichment_text": "",
        }

    issues: list[str] = []
    for step_id, err in result.errors.items():
        issues.append(f"Step {step_id}: {err}")

    return {
        "valid": len(issues) == 0,
        "element_count": len(result.groups),
        "issues": issues,
        "enrichment_text": result.enrichment_text,
    }

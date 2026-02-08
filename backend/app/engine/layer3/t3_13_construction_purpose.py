"""T3.13 — Construction Purpose Inference. ★

Infer why SVG is split into multiple sub-paths.
2 halves → half-fill, 3 pieces → fractional fill,
hole + outer → color independence.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.13",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T3.01", "T3.11"],
    description="Infer construction purpose of multi-subpath arrangement",
)
def construction_purpose(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    cmat = ctx.containment_matrix

    # Count nesting relationships
    nest_count = 0
    if cmat is not None:
        nest_count = int(cmat.sum())

    # Analyze arrangement
    purpose = "unknown"
    details = ""

    if n == 1:
        purpose = "single_element"
        details = "Single element, no construction analysis needed"
    elif n == 2:
        if nest_count > 0:
            purpose = "color_independence"
            details = "Inner + outer boundary: enables independent fill colors"
        else:
            # Check if they tile (halves)
            tiling = ctx.subpaths[0].features.get("overall_tiling", "SEPARATE")
            if tiling == "TILE":
                purpose = "half_fill"
                details = "2 tiling sub-paths: half-fill rendering"
            else:
                purpose = "composite"
                details = "2 separate elements composing a design"
    elif n == 3:
        if nest_count > 0:
            purpose = "layered_rendering"
            details = "Nested elements with independent fill layers"
        else:
            purpose = "fractional_fill"
            details = "3 sub-paths: fractional fill or tri-part composition"
    elif n <= 6:
        if nest_count > n:
            purpose = "complex_nesting"
            details = f"Complex nesting with {nest_count} containment relations"
        else:
            purpose = "multi_component"
            details = f"{n} components forming a composite icon"
    else:
        purpose = "complex_composition"
        details = f"{n} elements in complex arrangement"

    for sp in ctx.subpaths:
        sp.features["construction_purpose"] = purpose
        sp.features["construction_details"] = details

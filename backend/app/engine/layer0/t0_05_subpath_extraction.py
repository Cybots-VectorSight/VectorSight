"""T0.05 — Sub-Path Extraction.

Split compound paths (M...Z m...Z m...Z) into separate closed sub-paths.
Count sub-paths immediately — it's the first structural signal.

Note: The parser already handles basic sub-path extraction. This transform
records the structural metadata for downstream use.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T0.05",
    layer=Layer.PARSING,
    dependencies=["T0.04"],
    description="Record sub-path structure and count",
    tags={"always"},
)
def subpath_extraction(ctx: PipelineContext) -> None:
    total_subpaths = len(ctx.subpaths)

    for sp in ctx.subpaths:
        sp.features["total_elements"] = total_subpaths
        sp.features["is_closed"] = sp.closed
        sp.features["point_count"] = len(sp.points)

        # Classify segment composition
        seg_types = sp.features.get("segment_types", {})
        has_lines = seg_types.get("Line", 0) > 0
        has_curves = seg_types.get("CubicBezier", 0) > 0 or seg_types.get("QuadraticBezier", 0) > 0
        has_arcs = sp.features.get("has_arcs", False)

        if has_curves or has_arcs:
            if has_lines:
                sp.features["composition"] = "mixed"
            else:
                sp.features["composition"] = "curves"
        elif has_lines:
            sp.features["composition"] = "lines"
        else:
            sp.features["composition"] = "points"

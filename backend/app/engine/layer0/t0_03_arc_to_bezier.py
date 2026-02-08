"""T0.03 — Arc → Bézier Approximation.

Convert elliptical arc commands (A) to cubic bézier segments.
svgpathtools represents arcs as Arc objects; this records arc presence
and notes the conversion is handled by svgpathtools internally.
"""

from __future__ import annotations

from svgpathtools import Arc

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T0.03",
    layer=Layer.PARSING,
    dependencies=["T0.02"],
    description="Record arc segments (svgpathtools handles arc→bezier)",
    tags={"always"},
)
def arc_to_bezier(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        arc_count = sum(1 for seg in sp.segments if isinstance(seg, Arc))
        sp.features["arc_count"] = arc_count
        sp.features["has_arcs"] = arc_count > 0

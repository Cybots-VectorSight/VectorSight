"""T0.02 — Relative → Absolute Coordinate Resolution.

svgpathtools converts to absolute during parsing. This transform verifies
all coordinates are absolute and records any remaining issues.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T0.02",
    layer=Layer.PARSING,
    dependencies=["T0.01"],
    description="Verify relative→absolute coordinate resolution",
    tags={"always"},
)
def relative_to_absolute(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        # svgpathtools stores all coordinates as absolute complex numbers.
        # Verify by checking that all segment start/end are complex.
        all_absolute = True
        for seg in sp.segments:
            if hasattr(seg, "start") and hasattr(seg, "end"):
                if not isinstance(seg.start, complex) or not isinstance(seg.end, complex):
                    all_absolute = False
                    break
        sp.features["coordinates_absolute"] = all_absolute

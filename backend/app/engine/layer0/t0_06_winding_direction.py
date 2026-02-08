"""T0.06 — Winding Direction Detection.

Signed area via shoelace formula. Positive = CCW, Negative = CW.
Combined with fill-rule, determines which regions are inside vs outside.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import signed_area, winding_direction


@transform(
    id="T0.06",
    layer=Layer.PARSING,
    dependencies=["T0.04"],
    description="Detect winding direction via shoelace formula",
    tags={"always"},
)
def winding_direction_transform(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 3:
            sp.features["winding"] = "unknown"
            sp.features["signed_area"] = 0.0
            continue

        sa = signed_area(sp.points)
        wd = winding_direction(sp.points)

        sp.winding = wd
        sp.features["signed_area"] = sa
        sp.features["winding"] = "CCW" if wd > 0 else ("CW" if wd < 0 else "degenerate")

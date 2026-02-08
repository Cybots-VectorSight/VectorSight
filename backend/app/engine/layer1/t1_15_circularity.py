"""T1.15 — Circularity. ★

C = 4π·area/perimeter². Circle=1.0.
"""

from __future__ import annotations

import math

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.15",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T1.14"],
    description="Compute circularity (4π·area/perimeter²)",
)
def circularity(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        area = sp.area
        perimeter = sp.perimeter
        if perimeter > 1e-10:
            sp.features["circularity"] = round(4 * math.pi * area / (perimeter ** 2), 4)
        else:
            sp.features["circularity"] = 0.0

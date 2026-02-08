"""T1.16 — Rectangularity.

R = area / min_bounding_rectangle_area. Rectangle=1.0.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.16",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T1.14"],
    description="Compute rectangularity (area / bbox area)",
)
def rectangularity(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        bbox_area = sp.width * sp.height
        if bbox_area > 1e-10:
            sp.features["rectangularity"] = round(sp.area / bbox_area, 4)
        else:
            sp.features["rectangularity"] = 0.0

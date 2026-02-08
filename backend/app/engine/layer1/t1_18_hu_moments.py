"""T1.18 — Hu Moment Invariants. ★★

7 numbers invariant to translation, rotation, scale.
Primary use: shape similarity matching via DBSCAN clustering.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import hu_moments


@transform(
    id="T1.18",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute 7 Hu moment invariants (shape fingerprint)",
)
def hu_moments_transform(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 5:
            sp.features["hu_moments"] = [0.0] * 7
            continue

        moments = hu_moments(sp.points)
        sp.features["hu_moments"] = [round(float(m), 6) for m in moments]

"""T1.03 — Turning Function. ★★ PROVEN

Cumulative tangent angle θ(s) as function of arc-length.
Total ≈ 360° = closed loop, ≈ 180° = semicircle, ≈ 0° = straight.
Fastest open-vs-closed test.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import turning_function_total


@transform(
    id="T1.03",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute turning function total (open vs closed test)",
    tags={"always"},
)
def turning_function(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 3:
            sp.features["turning_total"] = 0.0
            sp.features["turning_classification"] = "degenerate"
            continue

        total = turning_function_total(sp.points)
        sp.features["turning_total"] = round(total, 1)

        # Classify
        if total > 340:
            sp.features["turning_classification"] = "closed_loop"
        elif total > 160:
            sp.features["turning_classification"] = "semicircle_or_arc"
        elif total > 20:
            sp.features["turning_classification"] = "curved"
        else:
            sp.features["turning_classification"] = "straight"

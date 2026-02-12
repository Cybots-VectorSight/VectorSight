"""T1.03 — Turning Function. ★★ PROVEN

Cumulative tangent angle θ(s) as function of arc-length.
Total ≈ 360° = closed loop, ≈ 180° = semicircle, ≈ 0° = straight.
Fastest open-vs-closed test.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import turning_function_total

# ── Turning function classification thresholds (degrees) ──
# A full closed loop has cumulative turning ≈ 360°.
# Allow 20° margin (360° - 20° = 340°) for imperfect closure.
_CLOSED_LOOP_DEG = 340.0   # ≈ 360° − 20° tolerance
# Semicircle/arc: > 160° ≈ 180° − 20° tolerance.
_SEMICIRCLE_DEG = 160.0    # ≈ 180° − 20° tolerance
# Curved: > 20° cumulative turn = meaningful curvature.
# Below 20° is essentially straight.
_CURVED_DEG = 20.0         # ≈ 360° / 18 (minimal curvature)


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
        if total > _CLOSED_LOOP_DEG:
            sp.features["turning_classification"] = "closed_loop"
        elif total > _SEMICIRCLE_DEG:
            sp.features["turning_classification"] = "semicircle_or_arc"
        elif total > _CURVED_DEG:
            sp.features["turning_classification"] = "curved"
        else:
            sp.features["turning_classification"] = "straight"

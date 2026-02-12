"""T2.07 — Figure-Ground Report. ★★★ CRITICAL

Report positive space, negative space, composite silhouette, and which carries
the primary design meaning.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.rasterizer import grid_fill_percentage

# ── Figure-ground classification thresholds ──
# Co-dominance: >30% fill (≈1/3) AND negative space = both carry meaning.
_CODOMINANCE_THRESHOLD = 30.0  # ≈ 1/3
# Positive-dominant: >50% fill (1/2) = positive space carries design.
_POSITIVE_DOMINANT_THRESHOLD = 50.0  # 1/2


@transform(
    id="T2.07",
    layer=Layer.VISUALIZATION,
    dependencies=["T2.05", "T2.06"],
    description="Generate figure-ground report",
)
def figure_ground(ctx: PipelineContext) -> None:
    if ctx.composite_grid is None:
        return

    pos_fill = grid_fill_percentage(ctx.composite_grid)
    neg_count = 0
    for sp in ctx.subpaths:
        neg_count = max(neg_count, sp.features.get("negative_space_count", 0))

    # Determine figure-ground relationship
    if neg_count > 0 and pos_fill > _CODOMINANCE_THRESHOLD:
        figure_ground_type = "both"
        description = "Both positive and negative space carry design meaning"
    elif pos_fill > _POSITIVE_DOMINANT_THRESHOLD:
        figure_ground_type = "positive"
        description = "Positive space (filled regions) carries the design"
    else:
        figure_ground_type = "distributed"
        description = "Design elements distributed across canvas"

    for sp in ctx.subpaths:
        sp.features["figure_ground_type"] = figure_ground_type
        sp.features["figure_ground_description"] = description
        sp.features["positive_fill_pct"] = round(pos_fill, 1)

"""T3.12 — Gap Analysis Between Sub-Paths. ★

For tiling sub-paths: count gap regions, measure areas,
check if gaps form recognizable pattern.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.morphology import connected_components_grid
from app.utils.rasterizer import invert_grid


@transform(
    id="T3.12",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T2.06", "T3.11"],
    description="Analyze gaps between tiling sub-paths",
)
def gap_analysis(ctx: PipelineContext) -> None:
    if ctx.composite_grid is None:
        return

    # Use negative space regions from T2.06
    neg_count = 0
    neg_regions = []
    for sp in ctx.subpaths:
        regions = sp.features.get("negative_space_regions", [])
        if regions:
            neg_count = len(regions)
            neg_regions = regions
            break

    # Compute total gap area
    total_gap_area_pct = sum(r.get("area_pct", 0) for r in neg_regions)

    # Check for pattern in gaps
    gap_pattern = "none"
    if neg_count >= 3:
        # Check if gap areas are similar (regular pattern)
        areas = [r.get("area_pct", 0) for r in neg_regions if r.get("area_pct", 0) > 0.5]
        if len(areas) >= 3:
            mean_area = np.mean(areas)
            if mean_area > 0:
                cv = float(np.std(areas) / mean_area)
                if cv < 0.3:
                    gap_pattern = "regular"
                else:
                    gap_pattern = "irregular"

    for sp in ctx.subpaths:
        sp.features["gap_count"] = neg_count
        sp.features["total_gap_area_pct"] = round(total_gap_area_pct, 1)
        sp.features["gap_pattern"] = gap_pattern

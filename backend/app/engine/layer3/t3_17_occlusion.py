"""T3.17 — Occlusion / Incompleteness Detection. ★

When arc < 360° AND missing portions face same direction → partially hidden.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform

# Full loop = 360 deg. Incomplete: < 360 - 60 = 300.
# 60 deg = 1/6 circle = sextant (minimum missing arc for "partial").
_INCOMPLETE_TURNING_DEG = 300  # 360 - 60
# 90% coverage = at most 36 deg gap (one compass direction missing).
_INCOMPLETE_COVERAGE_PCT = 90  # 100 * (1 - 1/10)
# Gap > 10% of perimeter (= 36/360 arc).
_OPEN_PATH_GAP_RATIO = 0.10
# Cardinal quadrants: 4 x 90 deg sectors centered on cardinal directions.
_QUAD_TOP = (45, 135)      # top sector
_QUAD_LEFT = (135, 225)    # left sector
_QUAD_BOTTOM = (225, 315)  # bottom sector
# Right: [315, 360) U [0, 45)


@transform(
    id="T3.17",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.06", "T1.03"],
    description="Detect partial occlusion or incompleteness in elements",
)
def occlusion(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        turning = sp.features.get("turning_total", 0)
        dir_coverage = sp.features.get("directional_coverage_pct", 100)
        gap_directions = sp.features.get("gap_directions", [])

        is_partial = False
        occlusion_dir = ""

        # Element is incomplete if:
        # 1. Not a full loop (turning < 300°)
        # 2. Directional coverage < 100% (has gaps)
        # 3. Gaps face a consistent direction
        if turning < _INCOMPLETE_TURNING_DEG and dir_coverage < _INCOMPLETE_COVERAGE_PCT:
            is_partial = True
            if gap_directions:
                occlusion_dir = gap_directions[0] if len(gap_directions) == 1 else "multiple"

        # Also check if it's an open path (not closed)
        if not sp.closed and len(sp.points) > 10:
            # Check if it's nearly closed (endpoints close together)
            start = sp.points[0]
            end = sp.points[-1]
            gap = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
            perimeter = sp.perimeter
            if perimeter > 0 and gap / perimeter > _OPEN_PATH_GAP_RATIO:
                is_partial = True
                # Direction of opening
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                angle = np.degrees(np.arctan2(-dy, dx)) % 360
                if _QUAD_TOP[0] <= angle < _QUAD_TOP[1]:
                    occlusion_dir = "top"
                elif _QUAD_LEFT[0] <= angle < _QUAD_LEFT[1]:
                    occlusion_dir = "left"
                elif _QUAD_BOTTOM[0] <= angle < _QUAD_BOTTOM[1]:
                    occlusion_dir = "bottom"
                else:
                    occlusion_dir = "right"

        sp.features["is_partial"] = is_partial
        sp.features["occlusion_direction"] = occlusion_dir

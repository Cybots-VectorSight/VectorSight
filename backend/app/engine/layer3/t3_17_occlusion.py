"""T3.17 — Occlusion / Incompleteness Detection. ★

When arc < 360° AND missing portions face same direction → partially hidden.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


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
        if turning < 300 and dir_coverage < 90:
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
            if perimeter > 0 and gap / perimeter > 0.1:
                is_partial = True
                # Direction of opening
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                angle = np.degrees(np.arctan2(-dy, dx)) % 360
                if 45 <= angle < 135:
                    occlusion_dir = "top"
                elif 135 <= angle < 225:
                    occlusion_dir = "left"
                elif 225 <= angle < 315:
                    occlusion_dir = "bottom"
                else:
                    occlusion_dir = "right"

        sp.features["is_partial"] = is_partial
        sp.features["occlusion_direction"] = occlusion_dir

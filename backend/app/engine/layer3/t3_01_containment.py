"""T3.01 — Containment Matrix. ★★★ CRITICAL

Build boolean containment matrix: containment[i][j] = True means shape i contains shape j.
Uses 5-point majority test (centroid + 4 bbox midpoints) with winding number.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import points_in_polygon_majority

# Container must have >= 50% area of contained element.
# Prevents thin lines from being "containers" of larger shapes.
_CONTAINMENT_MIN_AREA_RATIO = 0.5  # 1/2
# Progress reporting: only for > 10 elements (O(n^2) containment check).
_PROGRESS_THRESHOLD = 10  # 2x DOF for ellipse fitting


@transform(
    id="T3.01",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.14"],
    description="Build containment matrix using 5-point majority winding test",
)
def containment(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    matrix = np.zeros((n, n), dtype=bool)
    report_progress = ctx.progress_callback

    for i, sp_outer in enumerate(ctx.subpaths):
        if report_progress and n > _PROGRESS_THRESHOLD:
            report_progress(i / n)

        if sp_outer.polygon is None or sp_outer.polygon.is_empty:
            continue
        outer_pts = sp_outer.points
        if len(outer_pts) < 3:
            continue

        for j, sp_inner in enumerate(ctx.subpaths):
            if i == j:
                continue
            # Check if inner's test points are inside outer
            cx, cy = sp_inner.centroid
            bx0, by0, bx1, by1 = sp_inner.bbox
            test_points = [
                (cx, cy),
                ((bx0 + bx1) / 2, by0),  # top mid
                ((bx0 + bx1) / 2, by1),  # bottom mid
                (bx0, (by0 + by1) / 2),  # left mid
                (bx1, (by0 + by1) / 2),  # right mid
            ]
            if points_in_polygon_majority(test_points, outer_pts):
                # Also check area: container should be bigger
                if sp_outer.area > sp_inner.area * _CONTAINMENT_MIN_AREA_RATIO:
                    matrix[i][j] = True

    ctx.containment_matrix = matrix

    # Store per-element features
    for i, sp in enumerate(ctx.subpaths):
        contains = [ctx.subpaths[j].id for j in range(n) if matrix[i][j]]
        contained_by = [ctx.subpaths[j].id for j in range(n) if matrix[j][i]]
        sp.features["contains"] = contains
        sp.features["contained_by"] = contained_by
        sp.features["containment_depth"] = len(contained_by)

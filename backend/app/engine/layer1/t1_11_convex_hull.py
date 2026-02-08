"""T1.11 — Convex Hull & Convexity. ★★

Convexity ratio = shape_area / hull_area. Circle: 1.0, star: ~0.5.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.11",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute convex hull and convexity ratio",
)
def convex_hull(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 4:
            sp.features["convexity"] = 0.0
            sp.features["hull_area"] = 0.0
            continue

        try:
            hull = ConvexHull(sp.points)
            hull_area = float(hull.volume)  # In 2D, volume = area
            shape_area = sp.area

            if hull_area > 1e-10:
                sp.features["convexity"] = round(shape_area / hull_area, 4)
            else:
                sp.features["convexity"] = 0.0

            sp.features["hull_area"] = round(hull_area, 2)
            sp.features["hull_vertices"] = len(hull.vertices)
        except Exception:
            sp.features["convexity"] = 0.0
            sp.features["hull_area"] = 0.0

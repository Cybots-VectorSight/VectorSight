"""T1.14 — Basic Geometric Properties. ★★★

Area, perimeter, bounding box, centroid, aspect ratio, sub-path count,
segment type counts, orientation (PCA major axis angle), curvature variance.
Cheap, always compute.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import pca_orientation


@transform(
    id="T1.14",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04", "T0.05"],
    description="Compute basic geometric properties (area, perimeter, bbox, etc.)",
    tags={"always"},
)
def basic_geometric_props(ctx: PipelineContext) -> None:
    areas = []
    for sp in ctx.subpaths:
        sp.features["area"] = round(sp.area, 2)
        sp.features["perimeter"] = round(sp.perimeter, 2)
        sp.features["centroid_x"] = round(sp.centroid[0], 2)
        sp.features["centroid_y"] = round(sp.centroid[1], 2)
        sp.features["bbox_width"] = round(sp.width, 2)
        sp.features["bbox_height"] = round(sp.height, 2)

        # Aspect ratio
        if sp.height > 1e-10:
            sp.features["aspect_ratio"] = round(sp.width / sp.height, 3)
        else:
            sp.features["aspect_ratio"] = float("inf")

        # PCA orientation
        if len(sp.points) >= 3:
            sp.features["orientation"] = round(pca_orientation(sp.points), 1)
        else:
            sp.features["orientation"] = 0.0

        areas.append(sp.area)

    # Assign size tiers using natural breaks
    if areas:
        areas_arr = np.array(areas)
        if len(areas) >= 3:
            sorted_areas = np.sort(areas_arr)[::-1]
            # Simple tier assignment: top 20% = LARGE, bottom 30% = SMALL, rest = MEDIUM
            large_threshold = np.percentile(areas_arr[areas_arr > 0], 70) if np.any(areas_arr > 0) else 0
            small_threshold = np.percentile(areas_arr[areas_arr > 0], 30) if np.any(areas_arr > 0) else 0

            for sp in ctx.subpaths:
                a = sp.area
                if a >= large_threshold and a > 0:
                    sp.features["size_tier"] = "LARGE"
                elif a <= small_threshold or a == 0:
                    sp.features["size_tier"] = "SMALL"
                else:
                    sp.features["size_tier"] = "MEDIUM"
        else:
            for sp in ctx.subpaths:
                sp.features["size_tier"] = "MEDIUM"

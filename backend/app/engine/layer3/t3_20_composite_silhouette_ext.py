"""T3.20 — Composite Silhouette Extended Analysis. ★★

1. Rasterize all elements to binary grid.
2. Morphological close (bridge gaps).
3. Classify silhouette using T1.23 and T1.20 shape descriptors.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import centroid_distance_cv, pca_orientation
from app.utils.morphology import boundary_trace, morphological_close
from app.utils.rasterizer import grid_fill_percentage


@transform(
    id="T3.20",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T2.05"],
    description="Extended composite silhouette analysis with shape classification",
)
def composite_silhouette_ext(ctx: PipelineContext) -> None:
    if ctx.composite_grid is None:
        return

    closed = morphological_close(ctx.composite_grid, kernel_size=3)
    fill_pct = grid_fill_percentage(closed)

    # Compute silhouette shape properties
    boundary = boundary_trace(closed)
    silhouette_info: dict = {
        "fill_pct": round(fill_pct, 1),
        "bbox_aspect_ratio": round(ctx.canvas_width / max(ctx.canvas_height, 0.01), 2),
    }

    if boundary and len(boundary) >= 10:
        boundary_pts = np.array(boundary, dtype=np.float64)
        # Convert to canvas coords
        res = closed.shape[0]
        canvas_pts = np.zeros_like(boundary_pts)
        canvas_pts[:, 0] = boundary_pts[:, 1] / res * ctx.canvas_width
        canvas_pts[:, 1] = boundary_pts[:, 0] / res * ctx.canvas_height

        cv = centroid_distance_cv(canvas_pts)
        orientation = pca_orientation(canvas_pts)

        # Classify silhouette shape
        if cv < 0.1:
            sil_class = "circular"
        elif cv < 0.3:
            sil_class = "elliptical"
        elif fill_pct > 80:
            sil_class = "rectangular"
        else:
            sil_class = "organic"

        silhouette_info.update({
            "shape_class": sil_class,
            "centroid_cv": round(cv, 3),
            "orientation_deg": round(orientation, 1),
            "boundary_points": len(boundary),
        })

        # Convexity: compare fill with convex hull area approximation
        if ctx.composite_silhouette is not None:
            try:
                convex_area = ctx.composite_silhouette.convex_hull.area
                actual_area = ctx.composite_silhouette.area
                if convex_area > 0:
                    silhouette_info["convexity"] = round(actual_area / convex_area, 3)

                # Concavity locations
                diff = ctx.composite_silhouette.convex_hull.difference(ctx.composite_silhouette)
                if not diff.is_empty:
                    silhouette_info["has_concavities"] = True
                    silhouette_info["concavity_area_pct"] = round(diff.area / convex_area * 100, 1)
                else:
                    silhouette_info["has_concavities"] = False
            except Exception:
                pass

    for sp in ctx.subpaths:
        sp.features["composite_silhouette_ext"] = silhouette_info

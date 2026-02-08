"""T2.05 — Composite Silhouette Description. ★★★ CRITICAL

Trace outer boundary of all filled pixels combined (ignoring internal gaps).
Method: rasterize → morphological close → boundary trace → describe.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.morphology import boundary_trace, morphological_close
from app.utils.rasterizer import multi_element_grid


@transform(
    id="T2.05",
    layer=Layer.VISUALIZATION,
    dependencies=["T2.01"],
    description="Extract and describe composite silhouette",
)
def composite_silhouette(ctx: PipelineContext) -> None:
    if ctx.composite_grid is None:
        return

    # Morphological close to bridge small gaps
    closed = morphological_close(ctx.composite_grid, kernel_size=3)

    # Extract boundary
    boundary = boundary_trace(closed)
    if not boundary:
        return

    boundary_pts = np.array(boundary, dtype=np.float64)
    # Convert grid coords to canvas coords
    res = ctx.composite_grid.shape[0]
    canvas_pts = np.zeros_like(boundary_pts)
    canvas_pts[:, 0] = boundary_pts[:, 1] / res * ctx.canvas_width
    canvas_pts[:, 1] = boundary_pts[:, 0] / res * ctx.canvas_height

    # Build silhouette polygon
    try:
        from shapely.geometry import Polygon
        poly = Polygon(canvas_pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        ctx.composite_silhouette = poly
    except Exception:
        pass

    # Describe silhouette
    fill_pct = float(np.sum(closed)) / closed.size * 100
    for sp in ctx.subpaths:
        sp.features["composite_fill_pct"] = round(fill_pct, 1)

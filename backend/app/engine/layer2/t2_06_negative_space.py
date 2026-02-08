"""T2.06 — Negative Space Description. ★★★ CRITICAL

Analyze empty regions INSIDE composite bounding box.
Count connected components, describe each.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.morphology import connected_components_grid
from app.utils.rasterizer import invert_grid


@transform(
    id="T2.06",
    layer=Layer.VISUALIZATION,
    dependencies=["T2.01"],
    description="Analyze negative space (empty regions inside bbox)",
)
def negative_space(ctx: PipelineContext) -> None:
    if ctx.composite_grid is None:
        return

    inverted = invert_grid(ctx.composite_grid)

    # Find connected components of empty space
    labels, num_components = connected_components_grid(inverted)

    # Filter out the border component (largest empty region touching edges)
    edge_labels: set[int] = set()
    rows, cols = labels.shape
    for r in range(rows):
        if labels[r, 0] > 0:
            edge_labels.add(int(labels[r, 0]))
        if labels[r, cols - 1] > 0:
            edge_labels.add(int(labels[r, cols - 1]))
    for c in range(cols):
        if labels[0, c] > 0:
            edge_labels.add(int(labels[0, c]))
        if labels[rows - 1, c] > 0:
            edge_labels.add(int(labels[rows - 1, c]))

    internal_regions: list[dict] = []
    res = ctx.composite_grid.shape[0]

    for label_id in range(1, num_components + 1):
        if label_id in edge_labels:
            continue
        mask = labels == label_id
        pixel_count = int(np.sum(mask))
        if pixel_count < 2:
            continue

        # Get bounding box of this region
        ys, xs = np.where(mask)
        region_info = {
            "pixels": pixel_count,
            "bbox": (
                round(float(np.min(xs)) / res * ctx.canvas_width, 1),
                round(float(np.min(ys)) / res * ctx.canvas_height, 1),
                round(float(np.max(xs)) / res * ctx.canvas_width, 1),
                round(float(np.max(ys)) / res * ctx.canvas_height, 1),
            ),
            "area_pct": round(pixel_count / ctx.composite_grid.size * 100, 1),
        }
        internal_regions.append(region_info)

    for sp in ctx.subpaths:
        sp.features["negative_space_count"] = len(internal_regions)
        sp.features["negative_space_regions"] = internal_regions

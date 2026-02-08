"""T1.12 — Medial Axis / Skeleton. ★★ CONDITIONAL: FILLED SHAPES ONLY

For FILLED shapes: gives clean stick-figure topology.
For stroke-based SVGs: strokes already ARE the skeleton — skip.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.rasterizer import make_grid


@transform(
    id="T1.12",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute medial axis (skeleton) for filled shapes",
)
def medial_axis(ctx: PipelineContext) -> None:
    if ctx.is_stroke_based:
        for sp in ctx.subpaths:
            sp.features["medial_axis_junctions"] = 0
            sp.features["medial_axis_endpoints"] = 0
        return

    for sp in ctx.subpaths:
        if len(sp.points) < 10 or not sp.closed:
            sp.features["medial_axis_junctions"] = 0
            sp.features["medial_axis_endpoints"] = 0
            continue

        # Rasterize shape to small grid
        grid = make_grid(sp.points, ctx.canvas_width, ctx.canvas_height, resolution=32)

        try:
            from skimage.morphology import skeletonize

            skeleton = skeletonize(grid.astype(bool))
            # Count junctions (pixels with 3+ neighbors) and endpoints (1 neighbor)
            junctions = 0
            endpoints = 0
            rows, cols = skeleton.shape
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if not skeleton[r, c]:
                        continue
                    neighbors = int(np.sum(skeleton[r - 1 : r + 2, c - 1 : c + 2]) - 1)
                    if neighbors >= 3:
                        junctions += 1
                    elif neighbors == 1:
                        endpoints += 1

            sp.features["medial_axis_junctions"] = junctions
            sp.features["medial_axis_endpoints"] = endpoints
        except ImportError:
            sp.features["medial_axis_junctions"] = 0
            sp.features["medial_axis_endpoints"] = 0

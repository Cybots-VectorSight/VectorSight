"""T2.03 — Multi-Resolution Description. ★★

Describe at multiple zoom levels: COARSE (2×2) → MEDIUM (4×4) → FINE (8×8).
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.rasterizer import multi_element_grid, grid_fill_percentage

# Multi-resolution grid: octave doubling (standard in image processing).
_RES_COARSE = 4   # 2**2
_RES_MEDIUM = 8   # 2**3
_RES_FINE = 16     # 2**4


@transform(
    id="T2.03",
    layer=Layer.VISUALIZATION,
    dependencies=["T0.04"],
    description="Generate multi-resolution grid descriptions",
)
def multi_resolution(ctx: PipelineContext) -> None:
    all_points = [sp.points for sp in ctx.subpaths if len(sp.points) > 0]
    if not all_points:
        return

    resolutions = {"coarse": _RES_COARSE, "medium": _RES_MEDIUM, "fine": _RES_FINE}
    multi_res: dict[str, dict] = {}

    for name, res in resolutions.items():
        grid = multi_element_grid(all_points, ctx.canvas_width, ctx.canvas_height, res)
        fill_pct = grid_fill_percentage(grid)
        multi_res[name] = {
            "resolution": f"{res}x{res}",
            "fill_pct": round(fill_pct, 1),
        }

    for sp in ctx.subpaths:
        sp.features["multi_resolution"] = multi_res

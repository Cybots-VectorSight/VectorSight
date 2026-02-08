"""T2.01 — ASCII Grid. ★★★ PRIMARY — ALWAYS FIRST

Rasterize shapes onto text grid. Compute TWO grids:
1. POSITIVE grid: X = filled, . = empty
2. NEGATIVE grid (inverted): X = empty-inside-bbox, . = filled
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.rasterizer import grid_to_text, invert_grid, make_grid, multi_element_grid


@transform(
    id="T2.01",
    layer=Layer.VISUALIZATION,
    dependencies=["T0.04"],
    description="Generate positive and negative ASCII grids",
    tags={"always"},
)
def ascii_grid(ctx: PipelineContext) -> None:
    resolution = 32
    all_points = [sp.points for sp in ctx.subpaths if len(sp.points) > 0]

    if not all_points:
        return

    grid = multi_element_grid(all_points, ctx.canvas_width, ctx.canvas_height, resolution)
    ctx.composite_grid = grid
    ctx.ascii_grid_positive = grid_to_text(grid)
    ctx.ascii_grid_negative = grid_to_text(invert_grid(grid))

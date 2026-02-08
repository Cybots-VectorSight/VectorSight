"""T2.02 — Region Map. ★★

Divide canvas into 4×4 grid. Per region: fill percentage + density label.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.rasterizer import region_density


@transform(
    id="T2.02",
    layer=Layer.VISUALIZATION,
    dependencies=["T2.01"],
    description="Compute 4×4 region density map",
)
def region_map(ctx: PipelineContext) -> None:
    if ctx.composite_grid is None:
        return

    grid = ctx.composite_grid
    res = grid.shape[0]
    cell = res // 4
    regions: dict[str, dict] = {}

    labels = ["TL", "T", "TR", "R-TL", "L", "CL", "CR", "R",
              "R-BL", "BL", "BC", "BR", "B-BL", "B-L", "B-C", "B-R"]

    idx = 0
    for row in range(4):
        for col in range(4):
            r0, r1 = row * cell, (row + 1) * cell
            c0, c1 = col * cell, (col + 1) * cell
            density = region_density(grid, r0, r1, c0, c1)

            if density > 70:
                label = "dense"
            elif density > 30:
                label = "medium"
            elif density > 5:
                label = "sparse"
            else:
                label = "empty"

            key = f"R{row}{col}"
            regions[key] = {"density_pct": round(density, 1), "label": label}
            idx += 1

    ctx.region_map = regions

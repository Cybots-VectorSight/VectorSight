"""T3.04 — Relative Size Ratios. ★★

Compute area ratio between all element pairs.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.04",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.14"],
    description="Compute relative size ratios between elements",
)
def relative_size(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    areas = [sp.area for sp in ctx.subpaths]
    max_area = max(areas) if areas else 1.0
    if max_area < 1e-10:
        max_area = 1.0

    for i, sp in enumerate(ctx.subpaths):
        ratios = []
        for j in range(n):
            if i == j:
                continue
            if areas[j] > 1e-10:
                ratio = areas[i] / areas[j]
            else:
                ratio = float("inf")
            ratios.append({
                "element": ctx.subpaths[j].id,
                "size_ratio": round(ratio, 2),
            })

        sp.features["relative_sizes"] = ratios
        sp.features["area_rank"] = sorted(range(n), key=lambda k: -areas[k]).index(i) + 1
        sp.features["area_pct_of_largest"] = round(areas[i] / max_area * 100, 1)

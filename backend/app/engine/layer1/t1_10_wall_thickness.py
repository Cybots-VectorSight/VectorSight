"""T1.10 — Wall Thickness Profile. ★★★

For shapes with holes: distance between outer and inner boundaries at each angle.
Key discriminator (e.g., Bélo: top=1.7, sides=11.4, bottom=6.7).
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.10",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04", "T0.06"],
    description="Compute wall thickness for shapes with holes",
)
def wall_thickness(ctx: PipelineContext) -> None:
    # Find outer/inner pairs (opposite winding in same region)
    outer_inner_pairs: list[tuple[int, int]] = []
    for i, sp_i in enumerate(ctx.subpaths):
        for j, sp_j in enumerate(ctx.subpaths):
            if i >= j:
                continue
            if sp_i.winding != 0 and sp_j.winding != 0 and sp_i.winding != sp_j.winding:
                # Check bbox overlap
                xi1, yi1, xa1, ya1 = sp_i.bbox
                xi2, yi2, xa2, ya2 = sp_j.bbox
                if xi1 <= xa2 and xa1 >= xi2 and yi1 <= ya2 and ya1 >= yi2:
                    if sp_i.area > sp_j.area:
                        outer_inner_pairs.append((i, j))
                    else:
                        outer_inner_pairs.append((j, i))

    for outer_idx, inner_idx in outer_inner_pairs:
        outer = ctx.subpaths[outer_idx]
        inner = ctx.subpaths[inner_idx]

        if len(outer.points) < 5 or len(inner.points) < 5:
            continue

        # Compute wall thickness at 8 angular directions
        cx, cy = outer.centroid
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        thicknesses: list[float] = []

        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            # Find intersection with outer boundary
            outer_dist = _ray_intersection(cx, cy, dx, dy, outer.points)
            inner_dist = _ray_intersection(cx, cy, dx, dy, inner.points)
            if outer_dist is not None and inner_dist is not None:
                thicknesses.append(round(abs(outer_dist - inner_dist), 2))

        if thicknesses:
            outer.features["wall_thickness"] = thicknesses
            outer.features["wall_thickness_mean"] = round(float(np.mean(thicknesses)), 2)
            outer.features["wall_thickness_cv"] = round(
                float(np.std(thicknesses) / max(np.mean(thicknesses), 1e-10)), 3
            )
            outer.features["has_hole"] = True

    # Mark shapes without holes
    shapes_with_holes = {p[0] for p in outer_inner_pairs}
    for i, sp in enumerate(ctx.subpaths):
        if i not in shapes_with_holes:
            sp.features.setdefault("has_hole", False)


def _ray_intersection(
    cx: float, cy: float, dx: float, dy: float, points: np.ndarray
) -> float | None:
    """Find distance to first intersection of ray from (cx,cy) with polygon boundary."""
    min_dist = None
    for i in range(len(points) - 1):
        px, py = points[i]
        qx, qy = points[i + 1]

        # Ray-segment intersection
        denom = dx * (qy - py) - dy * (qx - px)
        if abs(denom) < 1e-10:
            continue

        t = ((qx - px) * (cy - py) - (qy - py) * (cx - px)) / denom
        u = (dx * (cy - py) - dy * (cx - px)) / denom

        if t > 0 and 0 <= u <= 1:
            if min_dist is None or t < min_dist:
                min_dist = t

    return min_dist

"""T3.05 — Relative Position Descriptions. ★★

Describe spatial relationship between each pair (e.g., "B is above A, offset 2 units right").
"""

from __future__ import annotations

import math

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.engine.spatial_constants import SPATIAL_JND_FRACTION


def _describe_position(dx: float, dy: float, diag: float) -> str:
    """Generate text description of relative position."""
    if diag < 1e-10:
        return "coincident"

    dist = math.sqrt(dx * dx + dy * dy)
    if dist < diag * SPATIAL_JND_FRACTION:
        return "coincident"

    parts = []
    # Vertical
    if abs(dy) > diag * SPATIAL_JND_FRACTION:
        if dy < 0:
            parts.append("above")
        else:
            parts.append("below")
    # Horizontal
    if abs(dx) > diag * SPATIAL_JND_FRACTION:
        if dx > 0:
            parts.append("right")
        else:
            parts.append("left")

    if not parts:
        return "coincident"

    return "-".join(parts)


@transform(
    id="T3.05",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.14"],
    description="Compute relative position descriptions between elements",
)
def relative_position(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    diag = ctx.viewbox_diagonal
    centroids = [sp.centroid for sp in ctx.subpaths]

    for i, sp in enumerate(ctx.subpaths):
        positions = []
        for j in range(n):
            if i == j:
                continue
            dx = centroids[j][0] - centroids[i][0]
            dy = centroids[j][1] - centroids[i][1]
            direction = _describe_position(dx, dy, diag)
            dist = math.sqrt(dx * dx + dy * dy)
            positions.append({
                "element": ctx.subpaths[j].id,
                "direction": direction,
                "offset_x": round(dx, 2),
                "offset_y": round(dy, 2),
                "distance": round(dist, 2),
            })
        sp.features["relative_positions"] = positions

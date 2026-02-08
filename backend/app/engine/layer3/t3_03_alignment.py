"""T3.03 — Alignment Detection. ★★

Detect horizontal, vertical, and center alignment between element centroids.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.03",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.14"],
    description="Detect alignment relationships between elements",
)
def alignment(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    diag = ctx.viewbox_diagonal
    tolerance = diag * 0.02  # 2% of diagonal

    centroids = [sp.centroid for sp in ctx.subpaths]

    h_groups: dict[float, list[int]] = {}
    v_groups: dict[float, list[int]] = {}

    # Group by x-alignment (vertical alignment line)
    for i, (cx, cy) in enumerate(centroids):
        placed = False
        for key in list(v_groups.keys()):
            if abs(cx - key) < tolerance:
                v_groups[key].append(i)
                placed = True
                break
        if not placed:
            v_groups[cx] = [i]

    # Group by y-alignment (horizontal alignment line)
    for i, (cx, cy) in enumerate(centroids):
        placed = False
        for key in list(h_groups.keys()):
            if abs(cy - key) < tolerance:
                h_groups[key].append(i)
                placed = True
                break
        if not placed:
            h_groups[cy] = [i]

    # Filter to groups with 2+ elements
    h_aligned = {k: v for k, v in h_groups.items() if len(v) >= 2}
    v_aligned = {k: v for k, v in v_groups.items() if len(v) >= 2}

    # Center alignment (both x and y match)
    center_aligned: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = centroids[i], centroids[j]
            if abs(ci[0] - cj[0]) < tolerance and abs(ci[1] - cj[1]) < tolerance:
                center_aligned.append((i, j))

    for i, sp in enumerate(ctx.subpaths):
        alignments = []
        for y_val, members in h_aligned.items():
            if i in members:
                others = [ctx.subpaths[m].id for m in members if m != i]
                if others:
                    alignments.append({"type": "horizontal", "with": others})
        for x_val, members in v_aligned.items():
            if i in members:
                others = [ctx.subpaths[m].id for m in members if m != i]
                if others:
                    alignments.append({"type": "vertical", "with": others})
        for pair in center_aligned:
            if i in pair:
                other = pair[1] if pair[0] == i else pair[0]
                alignments.append({"type": "center", "with": [ctx.subpaths[other].id]})

        sp.features["alignments"] = alignments
        sp.features["shares_center_x"] = any(a["type"] == "vertical" for a in alignments)
        sp.features["shares_center_y"] = any(a["type"] == "horizontal" for a in alignments)

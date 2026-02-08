"""T3.15 — Shared Center / Origin Detection. ★★

Test if element clusters share a common center point.
Fit circle to centroids and verify center match.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.15",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.14", "T3.07"],
    description="Detect elements sharing a common center or origin",
)
def shared_center(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    diag = ctx.viewbox_diagonal
    tolerance = diag * 0.03  # 3% of diagonal

    centroids = np.array([sp.centroid for sp in ctx.subpaths])

    # Check all pairs for shared center
    shared_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
            if dist < tolerance:
                shared_pairs.append((i, j))

    # Group elements sharing a center
    center_groups: list[list[int]] = []
    visited: set[int] = set()
    for i in range(n):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for pair in shared_pairs:
            if pair[0] == i and pair[1] not in visited:
                group.append(pair[1])
                visited.add(pair[1])
            elif pair[1] == i and pair[0] not in visited:
                group.append(pair[0])
                visited.add(pair[0])
        if len(group) >= 2:
            center_groups.append(group)

    # Check for concentric arrangement (shared center + different sizes)
    concentric_groups = []
    for group in center_groups:
        areas = [ctx.subpaths[g].area for g in group]
        if max(areas) > min(areas) * 1.5:  # Significant size difference
            sorted_members = sorted(group, key=lambda g: ctx.subpaths[g].area, reverse=True)
            concentric_groups.append({
                "members": [ctx.subpaths[g].id for g in sorted_members],
                "center": tuple(np.round(centroids[group[0]], 2)),
                "type": "concentric",
            })

    for i, sp in enumerate(ctx.subpaths):
        shares = [ctx.subpaths[j].id for pair in shared_pairs for j in pair if j != i and i in pair]
        sp.features["shares_center_with"] = shares
        sp.features["has_shared_center"] = len(shares) > 0
        sp.features["concentric_groups"] = concentric_groups

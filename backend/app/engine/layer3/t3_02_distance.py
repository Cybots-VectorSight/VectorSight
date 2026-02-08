"""T3.02 — Distance Matrix. ★★★ CRITICAL

Compute pairwise minimum distances between all element point-clouds.
Output closest points between shapes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.02",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T0.04"],
    description="Compute pairwise minimum distances between elements",
)
def distance_matrix(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    matrix = np.full((n, n), float("inf"))
    np.fill_diagonal(matrix, 0.0)

    # Build KD-trees for each element
    trees: list[cKDTree | None] = []
    for sp in ctx.subpaths:
        if len(sp.points) >= 2:
            trees.append(cKDTree(sp.points))
        else:
            trees.append(None)

    diag = ctx.viewbox_diagonal

    for i in range(n):
        if trees[i] is None:
            continue
        for j in range(i + 1, n):
            if trees[j] is None:
                continue
            # Query nearest neighbor from i to j
            dists, _ = trees[j].query(ctx.subpaths[i].points, k=1)
            min_dist = float(np.min(dists))
            matrix[i][j] = min_dist
            matrix[j][i] = min_dist

    ctx.distance_matrix = matrix

    # Store per-element: nearest neighbor and distance
    for i, sp in enumerate(ctx.subpaths):
        neighbors = []
        for j in range(n):
            if i == j:
                continue
            dist = float(matrix[i][j])
            if dist < float("inf"):
                neighbors.append({
                    "element": ctx.subpaths[j].id,
                    "distance": round(dist, 2),
                    "distance_pct": round(dist / diag * 100, 1) if diag > 0 else 0,
                })
        neighbors.sort(key=lambda x: x["distance"])
        sp.features["nearest_neighbors"] = neighbors[:5]  # Top 5
        if neighbors:
            sp.features["nearest_distance"] = neighbors[0]["distance"]
            sp.features["nearest_element"] = neighbors[0]["element"]

"""T3.18 — Connected Component Graph. ★★

Build graph: elements = nodes, edges = containment OR proximity < 2% of diagonal
OR shared boundary. Run connected components; each component is a visual unit.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


def _union_find_root(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union_find_merge(parent: list[int], rank: list[int], a: int, b: int) -> None:
    ra, rb = _union_find_root(parent, a), _union_find_root(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


@transform(
    id="T3.18",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T3.01", "T3.02"],
    description="Build connected component graph from containment and proximity",
)
def connected_components(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        if n == 1:
            ctx.component_labels = [0]
            ctx.subpaths[0].features["component_id"] = 0
        return

    cmat = ctx.containment_matrix
    dmat = ctx.distance_matrix
    diag = ctx.viewbox_diagonal
    proximity_threshold = diag * 0.02  # 2% of diagonal

    if cmat is None or dmat is None:
        return

    # Union-Find
    parent = list(range(n))
    rank = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            connected = False
            # Containment edge
            if cmat[i][j] or cmat[j][i]:
                connected = True
            # Proximity edge
            elif dmat[i][j] < proximity_threshold:
                connected = True

            if connected:
                _union_find_merge(parent, rank, i, j)

    # Normalize labels
    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    next_label = 0
    for i in range(n):
        root = _union_find_root(parent, i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels.append(root_to_label[root])

    ctx.component_labels = labels
    n_components = next_label

    # Build component summaries
    for i, sp in enumerate(ctx.subpaths):
        comp_id = labels[i]
        members = [ctx.subpaths[j].id for j in range(n) if labels[j] == comp_id]
        sp.features["component_id"] = comp_id
        sp.features["component_members"] = members
        sp.features["component_size"] = len(members)
        sp.features["total_components"] = n_components

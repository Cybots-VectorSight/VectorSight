"""T3.10 — Topology Report. ★★

Combine containment + proximity to describe topological patterns.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.10",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T3.01", "T3.02"],
    description="Combine containment and proximity into topological pattern description",
)
def topology(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    cmat = ctx.containment_matrix
    dmat = ctx.distance_matrix
    diag = ctx.viewbox_diagonal

    if cmat is None or dmat is None:
        return

    patterns: list[str] = []

    # 1. Nesting patterns
    for i in range(n):
        children = [j for j in range(n) if cmat[i][j]]
        if len(children) >= 2:
            child_classes = [ctx.subpaths[c].features.get("shape_class", "?") for c in children]
            patterns.append(
                f"{ctx.subpaths[i].id} contains {len(children)} elements "
                f"({', '.join(child_classes)})"
            )

    # 2. Touching / adjacent pairs (distance < 2% diagonal)
    touch_threshold = diag * 0.02
    touching_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if dmat[i][j] < touch_threshold and not cmat[i][j] and not cmat[j][i]:
                touching_pairs.append((i, j))

    if touching_pairs:
        patterns.append(f"{len(touching_pairs)} touching/adjacent pairs")

    # 3. Isolated elements (far from everything)
    far_threshold = diag * 0.25
    for i in range(n):
        min_dist = float("inf")
        for j in range(n):
            if i != j and dmat[i][j] < min_dist:
                min_dist = dmat[i][j]
        if min_dist > far_threshold:
            patterns.append(f"{ctx.subpaths[i].id} is spatially isolated")

    # 4. Overall topology label
    has_nesting = any(np.any(cmat[i]) for i in range(n))
    has_touching = len(touching_pairs) > 0

    if has_nesting and has_touching:
        topo_type = "nested+adjacent"
    elif has_nesting:
        topo_type = "nested"
    elif has_touching:
        topo_type = "adjacent"
    else:
        topo_type = "separated"

    for sp in ctx.subpaths:
        sp.features["topology_type"] = topo_type
        sp.features["topology_patterns"] = patterns

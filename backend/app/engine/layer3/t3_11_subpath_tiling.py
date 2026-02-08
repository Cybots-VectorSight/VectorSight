"""T3.11 — Sub-Path Tiling Classification. ★

Classify sub-path relationships: TILE / NEST / SEPARATE.
Check bbox overlaps, shared edges, vertex coincidence.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.11",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T3.01", "T3.02"],
    description="Classify sub-path tiling relationships",
)
def subpath_tiling(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    diag = ctx.viewbox_diagonal
    edge_threshold = diag * 0.01  # 1% of diagonal for shared edge detection

    cmat = ctx.containment_matrix
    dmat = ctx.distance_matrix
    if cmat is None or dmat is None:
        return

    for i, sp in enumerate(ctx.subpaths):
        classifications = []
        for j in range(n):
            if i == j:
                continue

            if cmat[i][j] or cmat[j][i]:
                rel = "NEST"
            elif dmat[i][j] < edge_threshold:
                # Check if they share boundary points (tiling)
                if len(sp.points) > 0 and len(ctx.subpaths[j].points) > 0:
                    tree = cKDTree(ctx.subpaths[j].points)
                    dists, _ = tree.query(sp.points, k=1)
                    shared_count = int(np.sum(dists < edge_threshold))
                    if shared_count >= 3:
                        rel = "TILE"
                    else:
                        rel = "TILE"  # Adjacent = tiling
                else:
                    rel = "TILE"
            else:
                rel = "SEPARATE"

            classifications.append({
                "element": ctx.subpaths[j].id,
                "relationship": rel,
            })

        sp.features["tiling_classifications"] = classifications

    # Overall classification
    tile_count = 0
    nest_count = 0
    for sp in ctx.subpaths:
        for c in sp.features.get("tiling_classifications", []):
            if c["relationship"] == "TILE":
                tile_count += 1
            elif c["relationship"] == "NEST":
                nest_count += 1

    if tile_count > nest_count:
        overall = "TILE"
    elif nest_count > 0:
        overall = "NEST"
    else:
        overall = "SEPARATE"

    for sp in ctx.subpaths:
        sp.features["overall_tiling"] = overall

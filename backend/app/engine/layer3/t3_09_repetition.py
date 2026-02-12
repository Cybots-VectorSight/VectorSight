"""T3.09 — Repetition Detection. ★★

Cluster elements by Hu moments to find repeated shapes.
Threshold derived via Otsu's method on pairwise Hu-moment distances.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform

# Fallback threshold when Otsu cannot be computed (<2 elements).
_HU_DIST_FALLBACK = 0.5


@transform(
    id="T3.09",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.18", "T1.23"],
    description="Detect repeated shapes via Hu moment clustering",
)
def repetition(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    # Collect Hu moments
    moments = []
    valid_indices = []
    for i, sp in enumerate(ctx.subpaths):
        hu = sp.features.get("hu_moments")
        if hu is not None and len(hu) == 7:
            moments.append(hu)
            valid_indices.append(i)

    if len(moments) < 2:
        return

    moment_arr = np.array(moments)
    # Pairwise distance in Hu-moment space
    dists = cdist(moment_arr, moment_arr, metric="euclidean")

    # Otsu's method splits Hu-moment distances into "same"/"different" classes.
    flat_dists = dists[np.triu_indices_from(dists, k=1)]
    if len(flat_dists) >= 2:
        try:
            threshold = threshold_otsu(flat_dists)
        except ValueError:
            threshold = float(np.median(flat_dists))
    else:
        threshold = _HU_DIST_FALLBACK
    visited = set()
    groups: list[list[int]] = []

    for i in range(len(valid_indices)):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, len(valid_indices)):
            if j in visited:
                continue
            if dists[i][j] < threshold:
                group.append(j)
                visited.add(j)
        if len(group) >= 2:
            groups.append(group)

    # Map back to subpath indices
    repetition_groups = []
    for group in groups:
        member_ids = [ctx.subpaths[valid_indices[g]].id for g in group]
        shape_class = ctx.subpaths[valid_indices[group[0]]].features.get("shape_class", "unknown")
        repetition_groups.append({
            "count": len(group),
            "members": member_ids,
            "shape_class": shape_class,
        })

    for i, sp in enumerate(ctx.subpaths):
        sp.features["repetition_groups"] = repetition_groups
        # Check if this element belongs to any repetition group
        sp.features["is_repeated"] = any(
            sp.id in g["members"] for g in repetition_groups
        )
        for g in repetition_groups:
            if sp.id in g["members"]:
                sp.features["repetition_count"] = g["count"]
                break
        else:
            sp.features["repetition_count"] = 1

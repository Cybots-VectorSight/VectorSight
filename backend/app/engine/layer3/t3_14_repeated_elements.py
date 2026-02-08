"""T3.14 — Repeated Element Detection. ★★

Cluster sub-paths by (perimeter, segment count, shape class).
If cluster ≥ 2: check for radial, grid, or mirror pattern.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.14",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.14", "T1.23", "T3.09"],
    description="Detect repeated elements and their arrangement pattern",
)
def repeated_elements(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    groups = ctx.subpaths[0].features.get("repetition_groups", [])
    if not groups:
        for sp in ctx.subpaths:
            sp.features["repeated_pattern"] = "none"
        return

    for group in groups:
        member_ids = group["members"]
        members = [sp for sp in ctx.subpaths if sp.id in member_ids]
        if len(members) < 2:
            continue

        centroids = np.array([m.centroid for m in members])
        center = centroids.mean(axis=0)

        # Check radial pattern: are elements equidistant from center?
        dists = np.sqrt(np.sum((centroids - center) ** 2, axis=1))
        if len(dists) > 0 and np.mean(dists) > 0:
            dist_cv = float(np.std(dists) / np.mean(dists)) if np.mean(dists) > 1e-6 else 1.0
        else:
            dist_cv = 1.0

        # Check grid pattern: regular spacing in x and y
        dx = np.diff(np.sort(centroids[:, 0]))
        dy = np.diff(np.sort(centroids[:, 1]))
        grid_x = len(dx) > 0 and np.std(dx) / (np.mean(dx) + 1e-10) < 0.2 if len(dx) > 1 else False
        grid_y = len(dy) > 0 and np.std(dy) / (np.mean(dy) + 1e-10) < 0.2 if len(dy) > 1 else False

        if dist_cv < 0.15 and len(members) >= 3:
            pattern = "radial"
        elif grid_x and grid_y:
            pattern = "grid"
        elif grid_x or grid_y:
            pattern = "linear_array"
        else:
            pattern = "irregular"

        group["pattern"] = pattern

    for sp in ctx.subpaths:
        sp.features["repetition_groups"] = groups
        for g in groups:
            if sp.id in g["members"]:
                sp.features["repeated_pattern"] = g.get("pattern", "none")
                break
        else:
            sp.features["repeated_pattern"] = "none"

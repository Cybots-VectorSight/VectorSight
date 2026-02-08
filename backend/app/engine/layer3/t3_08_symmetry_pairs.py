"""T3.08 — Symmetry Pair Detection. ★★

Identify mirror pairs from T1.20 symmetry data.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.08",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.20"],
    description="Detect mirror-symmetric element pairs",
)
def symmetry_pairs(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    # Use symmetry_pairs already computed by T1.20 (multi-element symmetry)
    pairs = ctx.symmetry_pairs
    axis = ctx.symmetry_axis
    score = ctx.symmetry_score

    for i, sp in enumerate(ctx.subpaths):
        mirror_partners = []
        for pair in pairs:
            if pair[0] == i:
                mirror_partners.append(ctx.subpaths[pair[1]].id)
            elif pair[1] == i:
                mirror_partners.append(ctx.subpaths[pair[0]].id)

        sp.features["mirror_partners"] = mirror_partners
        sp.features["has_mirror_pair"] = len(mirror_partners) > 0
        sp.features["global_symmetry_axis"] = axis
        sp.features["global_symmetry_score"] = round(score, 3)

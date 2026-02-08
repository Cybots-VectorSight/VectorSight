"""T4.01 — Canonical Orientation. ★

Rotate description to a canonical orientation so that
same shape in different rotations produces consistent features.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import pca_orientation


@transform(
    id="T4.01",
    layer=Layer.VALIDATION,
    dependencies=["T1.14"],
    description="Compute canonical orientation for rotation-invariant comparison",
)
def canonical_orientation(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 3:
            sp.features["canonical_orientation_deg"] = 0.0
            continue

        # PCA major axis angle
        angle = pca_orientation(sp.points)
        sp.features["canonical_orientation_deg"] = round(angle, 1)

        # Compute rotation to canonical (horizontal major axis)
        rotation_needed = -angle
        sp.features["rotation_to_canonical_deg"] = round(rotation_needed % 180, 1)

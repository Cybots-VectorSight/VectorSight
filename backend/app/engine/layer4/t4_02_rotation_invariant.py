"""T4.02 — Rotation-Invariant Feature Vector. ★

Build a feature vector that's invariant to rotation.
Uses Hu moments + Fourier descriptors + centroid distance CV.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T4.02",
    layer=Layer.VALIDATION,
    dependencies=["T1.18", "T1.05", "T1.04"],
    description="Build rotation-invariant feature vector for shape matching",
)
def rotation_invariant(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        hu = sp.features.get("hu_moments")
        fd = sp.features.get("fourier_magnitudes")
        cd_cv = sp.features.get("centroid_distance_cv", 0.0)
        circ = sp.features.get("circularity", 0.0)
        conv = sp.features.get("convexity", 0.0)

        # Build composite rotation-invariant vector
        components: list[float] = []
        if hu is not None:
            components.extend(float(h) for h in hu[:7])
        else:
            components.extend([0.0] * 7)

        if fd is not None:
            components.extend(float(f) for f in fd[:8])
        else:
            components.extend([0.0] * 8)

        components.append(float(cd_cv))
        components.append(float(circ))
        components.append(float(conv))

        sp.features["rotation_invariant_vector"] = [round(v, 6) for v in components]
        sp.features["rotation_invariant_dim"] = len(components)

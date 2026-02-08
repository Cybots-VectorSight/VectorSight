"""T1.01 — Curvature Profile. ★★★

Signed curvature κ(t) at each boundary point. THE shape fingerprint.
Summarize as: peak locations, magnitudes, smooth vs sharp, symmetry.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import curvature_at_points


@transform(
    id="T1.01",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute signed curvature profile along boundary",
)
def curvature_profile(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 5:
            sp.features["curvature_profile"] = []
            sp.features["curvature_mean"] = 0.0
            sp.features["curvature_std"] = 0.0
            sp.features["curvature_max"] = 0.0
            continue

        kappa = curvature_at_points(sp.points)
        sp.features["curvature_profile"] = kappa.tolist()
        sp.features["curvature_mean"] = float(np.mean(np.abs(kappa)))
        sp.features["curvature_std"] = float(np.std(kappa))
        sp.features["curvature_max"] = float(np.max(np.abs(kappa)))
        sp.features["curvature_variance"] = float(np.var(kappa))

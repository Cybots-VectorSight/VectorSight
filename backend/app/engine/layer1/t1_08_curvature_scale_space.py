"""T1.08 — Curvature Scale Space (CSS). ★★ PROVEN

Progressively smooth boundary, count inflection points at each scale.
Points persisting at high σ = fundamental features.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.08",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T1.01"],
    description="Curvature scale space — inflection persistence across smoothing",
)
def curvature_scale_space(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        kappa = sp.features.get("curvature_profile", [])
        if len(kappa) < 10:
            sp.features["css_scales"] = []
            sp.features["css_max_persistent"] = 0
            continue

        kappa_arr = np.array(kappa)
        scales = [1, 2, 4, 8, 16]
        inflection_counts: list[int] = []

        for sigma in scales:
            if sigma >= len(kappa_arr) / 2:
                break
            smoothed = gaussian_filter1d(kappa_arr, sigma=sigma)
            signs = np.sign(smoothed)
            changes = int(np.sum(np.abs(np.diff(signs)) > 0))
            inflection_counts.append(changes)

        sp.features["css_scales"] = inflection_counts
        # Inflections at highest sigma = fundamental shape features
        sp.features["css_max_persistent"] = inflection_counts[-1] if inflection_counts else 0

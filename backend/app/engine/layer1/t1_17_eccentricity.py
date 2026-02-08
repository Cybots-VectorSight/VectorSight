"""T1.17 — Eccentricity.

Major/minor axis ratio. Circle=1.0. Also gives principal axis orientation.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.17",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute eccentricity (major/minor axis ratio)",
)
def eccentricity(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 5:
            sp.features["eccentricity"] = 1.0
            continue

        centered = sp.points - np.mean(sp.points, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]

        if eigenvalues[-1] > 1e-10:
            ratio = float(np.sqrt(eigenvalues[0] / eigenvalues[-1]))
            sp.features["eccentricity"] = round(ratio, 3)
        else:
            sp.features["eccentricity"] = float("inf")

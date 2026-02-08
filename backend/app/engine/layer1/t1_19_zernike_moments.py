"""T1.19 — Zernike Moments. ★ INTERNAL

Orthogonal complex moment basis. More discriminating than Hu. Internal.
Simplified approximation using radial distance histogram.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.19",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute Zernike moment approximation (internal)",
)
def zernike_moments(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 10:
            sp.features["zernike_moments"] = []
            continue

        # Map points to unit disk
        cx, cy = sp.centroid
        pts = sp.points - np.array([cx, cy])
        max_r = np.max(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2))
        if max_r < 1e-10:
            sp.features["zernike_moments"] = []
            continue

        pts = pts / max_r  # normalize to unit disk
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        theta = np.arctan2(pts[:, 1], pts[:, 0])

        # Compute first few Zernike-like moments (radial polynomial × angular)
        moments: list[float] = []
        for n in range(5):
            for m in range(-n, n + 1, 2):
                if abs(m) > n:
                    continue
                # Simplified: use r^n * exp(i*m*theta)
                z = r ** n * np.exp(1j * m * theta)
                moment = float(np.abs(np.mean(z)))
                moments.append(round(moment, 6))

        sp.features["zernike_moments"] = moments[:10]  # cap at 10

"""T1.19 — Zernike Moments. ★ INTERNAL

Orthogonal complex moment basis. More discriminating than Hu. Internal.
Simplified approximation using radial distance histogram.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform

# ── Zernike moment constants ──
# Minimum points for stable moment computation: 2× ellipse DOF = 10.
_MIN_POINTS = 10
# Machine-epsilon for max radial distance (below = degenerate).
_RADIUS_EPSILON = 1e-10
# Max order n for Zernike polynomial basis.
# Order 4 (n=0..4) yields up to 9 moments — captures shape features
# up to 4-fold symmetry (sufficient for most icon discrimination).
_MAX_ORDER = 5  # range(5) → n = 0, 1, 2, 3, 4
# Output cap: max moments to store per element (keeps features compact).
_MAX_MOMENTS = 10


@transform(
    id="T1.19",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute Zernike moment approximation (internal)",
)
def zernike_moments(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < _MIN_POINTS:
            sp.features["zernike_moments"] = []
            continue

        # Map points to unit disk
        cx, cy = sp.centroid
        pts = sp.points - np.array([cx, cy])
        max_r = np.max(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2))
        if max_r < _RADIUS_EPSILON:
            sp.features["zernike_moments"] = []
            continue

        pts = pts / max_r  # normalize to unit disk
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        theta = np.arctan2(pts[:, 1], pts[:, 0])

        # Compute first few Zernike-like moments (radial polynomial × angular)
        moments: list[float] = []
        for n in range(_MAX_ORDER):
            for m in range(-n, n + 1, 2):
                if abs(m) > n:
                    continue
                # Simplified: use r^n * exp(i*m*theta)
                z = r ** n * np.exp(1j * m * theta)
                moment = float(np.abs(np.mean(z)))
                moments.append(round(moment, 6))

        sp.features["zernike_moments"] = moments[:_MAX_MOMENTS]

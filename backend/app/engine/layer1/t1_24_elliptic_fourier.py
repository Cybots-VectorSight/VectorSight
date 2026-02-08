"""T1.24 — Elliptic Fourier Descriptors (EFD). ★ INTERNAL

Decompose closed contour into frequency components.
Low-frequency = gross shape, high = fine detail.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.24",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute elliptic Fourier descriptors (internal)",
)
def elliptic_fourier(ctx: PipelineContext) -> None:
    n_harmonics = 12

    for sp in ctx.subpaths:
        if len(sp.points) < 10 or not sp.closed:
            sp.features["efd"] = []
            continue

        pts = sp.points
        # Compute EFD
        dx = np.diff(pts[:, 0])
        dy = np.diff(pts[:, 1])
        dt = np.sqrt(dx ** 2 + dy ** 2)
        dt = np.where(dt < 1e-10, 1e-10, dt)
        t = np.concatenate([[0], np.cumsum(dt)])
        T = t[-1]

        if T < 1e-10:
            sp.features["efd"] = []
            continue

        coeffs: list[list[float]] = []
        for n in range(1, n_harmonics + 1):
            factor = T / (2 * n * n * np.pi * np.pi)
            cos_diff = np.cos(2 * n * np.pi * t[1:] / T) - np.cos(2 * n * np.pi * t[:-1] / T)
            sin_diff = np.sin(2 * n * np.pi * t[1:] / T) - np.sin(2 * n * np.pi * t[:-1] / T)

            an = factor * np.sum((dx / dt) * cos_diff)
            bn = factor * np.sum((dx / dt) * sin_diff)
            cn = factor * np.sum((dy / dt) * cos_diff)
            dn = factor * np.sum((dy / dt) * sin_diff)

            coeffs.append([round(float(v), 6) for v in [an, bn, cn, dn]])

        sp.features["efd"] = coeffs

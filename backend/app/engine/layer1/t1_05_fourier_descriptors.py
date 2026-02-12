"""T1.05 — Fourier Shape Descriptors. ★ INTERNAL ONLY

FFT of boundary coordinates. Internal fingerprint for shape comparison/matching.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform

# ── Fourier descriptor constants ──
# Minimum points for stable FFT: 2× ellipse DOF = 10.
_MIN_POINTS = 10
# Machine-epsilon for DC component magnitude.
_DC_EPSILON = 1e-10
# Max harmonics: 16 = 2^4. Nyquist: N/2 harmonics from N samples,
# but only first 16 carry shape information (higher = noise).
_MAX_HARMONICS = 16


@transform(
    id="T1.05",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute Fourier shape descriptors (internal fingerprint)",
)
def fourier_descriptors(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < _MIN_POINTS:
            sp.features["fourier_descriptors"] = []
            continue

        # Convert to complex representation
        z = sp.points[:, 0] + 1j * sp.points[:, 1]
        # Center
        z = z - np.mean(z)
        # FFT
        fft = np.fft.fft(z)
        # Normalize by DC component
        if abs(fft[0]) > _DC_EPSILON:
            fft = fft / abs(fft[0])
        # Take magnitude of first N harmonics
        n_harmonics = min(_MAX_HARMONICS, len(fft) // 2)
        magnitudes = np.abs(fft[1 : n_harmonics + 1])
        sp.features["fourier_descriptors"] = [round(float(m), 6) for m in magnitudes]

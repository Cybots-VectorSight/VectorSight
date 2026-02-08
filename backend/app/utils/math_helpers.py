"""Math helpers — CV, PCA, log-transform, etc. No engine imports."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def coefficient_of_variation(values: NDArray[np.float64]) -> float:
    """CV = std / mean. Used for circularity detection."""
    mean = float(np.mean(values))
    if abs(mean) < 1e-10:
        return float("inf")
    return float(np.std(values) / mean)


def log_transform(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """sign(v) * log10(1 + |v|). Used for Hu moment comparison."""
    return np.sign(values) * np.log10(1 + np.abs(values))


def snap_to_clean(value: float, multiples: list[float], tolerance: float = 1.0) -> float:
    """Snap a value to the nearest clean multiple if within tolerance.

    Used for angle snapping: 45.4° → 45° when tolerance=1.0°.
    """
    for m in multiples:
        if m == 0:
            continue
        nearest = round(value / m) * m
        if abs(value - nearest) <= tolerance:
            return nearest
    return value


CLEAN_ANGLES = [15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0, 270.0, 360.0]


def snap_angle(degrees: float, tolerance: float = 1.0) -> float:
    """Snap an angle to the nearest clean angle if within tolerance."""
    for clean in CLEAN_ANGLES:
        if abs(degrees - clean) <= tolerance:
            return clean
    return degrees

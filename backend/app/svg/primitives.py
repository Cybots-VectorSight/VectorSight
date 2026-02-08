"""Circle/rect detection from cubic bezier sequences."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.utils.geometry import centroid_distance_cv


def is_circle_from_cubics(points: NDArray[np.float64], threshold: float = 0.1) -> bool:
    """Detect if sampled points form a circle. CV < threshold = circular."""
    if len(points) < 10:
        return False
    return centroid_distance_cv(points) < threshold


def is_line_from_cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    tolerance: float = 0.01,
) -> bool:
    """Detect if a cubic bezier is actually a straight line (control points collinear)."""
    pts = np.array([p0, p1, p2, p3])
    if len(pts) < 4:
        return False

    # Check collinearity: cross product of vectors should be near zero
    v1 = pts[3] - pts[0]
    v2 = pts[1] - pts[0]
    v3 = pts[2] - pts[0]

    length = np.linalg.norm(v1)
    if length < 1e-10:
        return True

    cross1 = abs(v1[0] * v2[1] - v1[1] * v2[0]) / length
    cross2 = abs(v1[0] * v3[1] - v1[1] * v3[0]) / length

    return cross1 < tolerance and cross2 < tolerance

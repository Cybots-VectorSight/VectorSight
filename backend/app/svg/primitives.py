"""Circle/rect detection from cubic bezier sequences."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.utils.geometry import centroid_distance_cv

# CV < 0.10 = centroid distances essentially constant -> circle.
# CV = sigma/mu: relative SD < 10% means within 1 SD at n=100.
_CV_CIRCULAR = 0.10

# Minimum points for stable circle detection: 2x ellipse DOF (5) = 10.
_MIN_CIRCLE_POINTS = 10


def is_circle_from_cubics(points: NDArray[np.float64], threshold: float = _CV_CIRCULAR) -> bool:
    """Detect if sampled points form a circle. CV < threshold = circular."""
    if len(points) < _MIN_CIRCLE_POINTS:
        return False
    return centroid_distance_cv(points) < threshold


# Collinearity tolerance: 1% of segment length.
# Sub-pixel for typical SVG viewport sizes (100-1000px).
_LINE_COLLINEARITY_TOL = 0.01


def is_line_from_cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    tolerance: float = _LINE_COLLINEARITY_TOL,
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

"""Leaf-node geometry helpers. No engine imports."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def signed_area(points: NDArray[np.float64]) -> float:
    """Shoelace formula for signed area. Positive = CCW, Negative = CW."""
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def winding_direction(points: NDArray[np.float64]) -> int:
    """Return 1 for CCW, -1 for CW, 0 if degenerate."""
    sa = signed_area(points)
    if sa > 0:
        return 1
    elif sa < 0:
        return -1
    return 0


def bbox(points: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Compute (xmin, ymin, xmax, ymax) bounding box."""
    if len(points) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def centroid(points: NDArray[np.float64]) -> tuple[float, float]:
    """Compute centroid of a point set."""
    if len(points) == 0:
        return (0.0, 0.0)
    return (float(np.mean(points[:, 0])), float(np.mean(points[:, 1])))


def centroid_distances(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance from centroid to each boundary point."""
    cx, cy = centroid(points)
    return np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)


def centroid_distance_cv(points: NDArray[np.float64]) -> float:
    """Coefficient of variation of centroid distances. <0.1 = circular."""
    dists = centroid_distances(points)
    if len(dists) == 0:
        return float("inf")
    mean = float(np.mean(dists))
    if mean < 1e-10:
        return float("inf")
    return float(np.std(dists) / mean)


def arc_lengths(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative arc-length along a point sequence."""
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def tangent_angles(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tangent angle at each point (atan2 of forward difference)."""
    diffs = np.diff(points, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    return angles


def turning_function_total(points: NDArray[np.float64]) -> float:
    """Total turning angle along path. ~360° = closed loop, ~180° = semicircle."""
    angles = tangent_angles(points)
    if len(angles) < 2:
        return 0.0
    # Unwrap angle differences
    diffs = np.diff(angles)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return float(np.abs(np.sum(diffs)) * 180 / np.pi)


def winding_number(point: tuple[float, float], polygon_points: NDArray[np.float64]) -> int:
    """Compute winding number of point w.r.t. polygon boundary.

    Non-zero → point is inside polygon.
    """
    px, py = point
    x = polygon_points[:, 0]
    y = polygon_points[:, 1]
    n = len(x)

    wn = 0
    for i in range(n - 1):
        if y[i] <= py:
            if y[i + 1] > py:
                # Upward crossing
                cross = (x[i + 1] - x[i]) * (py - y[i]) - (px - x[i]) * (y[i + 1] - y[i])
                if cross > 0:
                    wn += 1
        else:
            if y[i + 1] <= py:
                # Downward crossing
                cross = (x[i + 1] - x[i]) * (py - y[i]) - (px - x[i]) * (y[i + 1] - y[i])
                if cross < 0:
                    wn -= 1
    return wn


def point_in_polygon(
    point: tuple[float, float],
    polygon_points: NDArray[np.float64],
    n_test_points: int = 5,
) -> bool:
    """Test containment using winding number. For robustness, test centroid + bbox midpoints."""
    return winding_number(point, polygon_points) != 0


def points_in_polygon_majority(
    test_points: list[tuple[float, float]],
    polygon_points: NDArray[np.float64],
) -> bool:
    """Majority vote: True if >50% of test points are inside polygon."""
    inside = sum(1 for p in test_points if winding_number(p, polygon_points) != 0)
    return inside > len(test_points) / 2


def hu_moments(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute 7 Hu moment invariants from a point set (binary image approximation).

    Uses log-transform: sign(h) * log10(1 + |h|) for distance computation.
    """
    if len(points) < 3:
        return np.zeros(7)

    # Center the points
    cx, cy = centroid(points)
    x = points[:, 0] - cx
    y = points[:, 1] - cy

    # Raw moments
    def mu(p: int, q: int) -> float:
        return float(np.sum(x**p * y**q))

    m00 = float(len(points))
    if m00 < 1:
        return np.zeros(7)

    # Normalized central moments
    def eta(p: int, q: int) -> float:
        return mu(p, q) / (m00 ** (1 + (p + q) / 2))

    e20 = eta(2, 0)
    e02 = eta(0, 2)
    e11 = eta(1, 1)
    e30 = eta(3, 0)
    e03 = eta(0, 3)
    e21 = eta(2, 1)
    e12 = eta(1, 2)

    # Hu invariants
    h1 = e20 + e02
    h2 = (e20 - e02) ** 2 + 4 * e11**2
    h3 = (e30 - 3 * e12) ** 2 + (3 * e21 - e03) ** 2
    h4 = (e30 + e12) ** 2 + (e21 + e03) ** 2
    h5 = (e30 - 3 * e12) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2) + (
        3 * e21 - e03
    ) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2)
    h6 = (e20 - e02) * ((e30 + e12) ** 2 - (e21 + e03) ** 2) + 4 * e11 * (e30 + e12) * (
        e21 + e03
    )
    h7 = (3 * e21 - e03) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2) - (
        e30 - 3 * e12
    ) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2)

    raw = np.array([h1, h2, h3, h4, h5, h6, h7])
    # Log-transform
    return np.sign(raw) * np.log10(1 + np.abs(raw))


def pca_orientation(points: NDArray[np.float64]) -> float:
    """Major axis angle from PCA (degrees, 0-180)."""
    if len(points) < 3:
        return 0.0
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle = float(np.arctan2(major_axis[1], major_axis[0]) * 180 / np.pi)
    return angle % 180


def curvature_at_points(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed curvature at each interior point using finite differences.

    κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    """
    if len(points) < 3:
        return np.array([])

    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa

"""Contour extraction — marching squares, RDP simplification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rdp_simplify(
    points: NDArray[np.float64],
    epsilon: float,
) -> NDArray[np.float64]:
    """Ramer-Douglas-Peucker line simplification.

    Reduces point count while preserving shape within epsilon tolerance.
    """
    if len(points) <= 2:
        return points

    # Find the point with maximum distance from the line (start, end)
    start = points[0]
    end = points[-1]

    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        # All points at same location
        return points[[0, -1]]

    line_unit = line_vec / line_len

    # Perpendicular distance from each point to the line
    vecs = points - start
    projections = np.dot(vecs, line_unit)
    closest = start + np.outer(projections, line_unit)
    distances = np.linalg.norm(points - closest, axis=1)

    max_idx = int(np.argmax(distances))
    max_dist = distances[max_idx]

    if max_dist > epsilon:
        # Recurse
        left = rdp_simplify(points[: max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return points[[0, -1]]


def extract_contour_from_grid(
    grid: NDArray[np.int8],
    canvas_w: float,
    canvas_h: float,
) -> NDArray[np.float64]:
    """Extract outer contour from binary grid, returning canvas-space coordinates.

    Simple boundary trace approach (not full marching squares).
    """
    from app.utils.morphology import boundary_trace

    boundary = boundary_trace(grid)
    if not boundary:
        return np.empty((0, 2))

    resolution = grid.shape[0]
    points = np.array(boundary, dtype=np.float64)

    # Convert grid coords to canvas coords
    points[:, 0] = points[:, 0] / resolution * canvas_h
    points[:, 1] = points[:, 1] / resolution * canvas_w

    # Swap to (x, y) from (row, col)
    points = points[:, ::-1]

    # Sort by angle from centroid for ordered contour
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    order = np.argsort(angles)
    points = points[order]

    return points

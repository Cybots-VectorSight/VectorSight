"""Rasterization utilities — SVG to pixel grid, grid to text.

Adapted from demo_grid.py POC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_grid(
    points: NDArray[np.float64] | list[tuple[float, float]],
    canvas_w: float,
    canvas_h: float,
    resolution: int = 32,
) -> NDArray[np.int8]:
    """Rasterize points onto a resolution×resolution grid.

    Args:
        points: Nx2 array or list of (x, y) coordinates.
        canvas_w: Canvas width (from viewBox).
        canvas_h: Canvas height (from viewBox).
        resolution: Grid size (default 32×32).

    Returns:
        Grid array where 1 = filled, 0 = empty.
    """
    grid = np.zeros((resolution, resolution), dtype=np.int8)
    pts = np.asarray(points)

    if len(pts) == 0:
        return grid

    cols = ((pts[:, 0] / canvas_w) * resolution).astype(int)
    rows = ((pts[:, 1] / canvas_h) * resolution).astype(int)

    # Clamp to grid bounds
    cols = np.clip(cols, 0, resolution - 1)
    rows = np.clip(rows, 0, resolution - 1)

    grid[rows, cols] = 1
    return grid


def grid_to_text(
    grid: NDArray[np.int8],
    filled: str = "X",
    empty: str = ".",
) -> str:
    """Convert a grid to a text representation."""
    rows = []
    for row in grid:
        rows.append(" ".join(filled if cell else empty for cell in row))
    return "\n".join(rows)


def invert_grid(grid: NDArray[np.int8]) -> NDArray[np.int8]:
    """Invert grid (swap filled/empty). For negative space analysis."""
    return (1 - grid).astype(np.int8)


def grid_fill_percentage(grid: NDArray[np.int8]) -> float:
    """Percentage of filled cells."""
    total = grid.size
    if total == 0:
        return 0.0
    return float(np.sum(grid) / total * 100)


def multi_element_grid(
    element_points: list[NDArray[np.float64]],
    canvas_w: float,
    canvas_h: float,
    resolution: int = 32,
) -> NDArray[np.int8]:
    """Rasterize multiple elements onto a single grid."""
    grid = np.zeros((resolution, resolution), dtype=np.int8)
    for pts in element_points:
        grid |= make_grid(pts, canvas_w, canvas_h, resolution)
    return grid


def region_density(
    grid: NDArray[np.int8],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> float:
    """Fill percentage for a sub-region of the grid."""
    region = grid[row_start:row_end, col_start:col_end]
    total = region.size
    if total == 0:
        return 0.0
    return float(np.sum(region) / total * 100)

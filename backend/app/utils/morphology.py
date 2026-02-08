"""Morphological operations for composite silhouette extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def morphological_close(grid: NDArray[np.int8], kernel_size: int = 3) -> NDArray[np.int8]:
    """Binary morphological close (dilate then erode) to bridge small gaps.

    Uses a simple square structuring element.
    """
    dilated = _dilate(grid, kernel_size)
    closed = _erode(dilated, kernel_size)
    return closed


def _dilate(grid: NDArray[np.int8], kernel_size: int) -> NDArray[np.int8]:
    """Binary dilation with square kernel."""
    pad = kernel_size // 2
    padded = np.pad(grid, pad, mode="constant", constant_values=0)
    result = np.zeros_like(grid)
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            region = padded[r : r + kernel_size, c : c + kernel_size]
            result[r, c] = 1 if np.any(region) else 0
    return result.astype(np.int8)


def _erode(grid: NDArray[np.int8], kernel_size: int) -> NDArray[np.int8]:
    """Binary erosion with square kernel."""
    pad = kernel_size // 2
    padded = np.pad(grid, pad, mode="constant", constant_values=0)
    result = np.zeros_like(grid)
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            region = padded[r : r + kernel_size, c : c + kernel_size]
            result[r, c] = 1 if np.all(region) else 0
    return result.astype(np.int8)


def boundary_trace(grid: NDArray[np.int8]) -> list[tuple[int, int]]:
    """Extract outer boundary pixels from a binary grid.

    Returns list of (row, col) coordinates of boundary pixels.
    """
    rows, cols = grid.shape
    boundary: list[tuple[int, int]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0:
                continue
            # Check if any neighbor is empty (4-connected)
            is_boundary = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    is_boundary = True
                    break
                if grid[nr, nc] == 0:
                    is_boundary = True
                    break
            if is_boundary:
                boundary.append((r, c))

    return boundary


def connected_components_grid(grid: NDArray[np.int8]) -> tuple[NDArray[np.int32], int]:
    """Label connected components in a binary grid using flood fill (4-connected)."""
    rows, cols = grid.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    current_label = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1 and labels[r, c] == 0:
                current_label += 1
                _flood_fill(grid, labels, r, c, current_label)

    return labels, current_label


def _flood_fill(
    grid: NDArray[np.int8],
    labels: NDArray[np.int32],
    start_r: int,
    start_c: int,
    label: int,
) -> None:
    """BFS flood fill for connected component labeling."""
    rows, cols = grid.shape
    queue = [(start_r, start_c)]
    labels[start_r, start_c] = label

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1 and labels[nr, nc] == 0:
                labels[nr, nc] = label
                queue.append((nr, nc))

"""Rasterization utilities — SVG to pixel grid, grid to text, silhouette descriptors.

Adapted from demo_grid.py POC.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from shapely.geometry import Polygon

# ── Named constants — zero magic numbers ──

# Center tolerance: ±15% from grid midpoint = center zone [35%,65%].
# Matches standard thirds grid: left [0%,33%], center [33%,67%], right.
_CENTER_TOLERANCE_PCT = 0.15

# Minimum filled pixels for stable centroid: 2× the 5 DOF for ellipse fitting.
_MIN_FILLED_FOR_CENTROID = 10

# Short branch: <30% of main axis length (≈1/3).
_SHORT_BRANCH_RATIO = 0.30

# Weber's law JND: ~15-20% for spatial contrast. Peaks below this
# fraction of the min-max range are not perceptually distinct.
_PEAK_PROMINENCE_FRACTION = 0.20

# Noise margin: 5% above mean distance to avoid false peaks from aliasing.
_PEAK_NOISE_MARGIN = 1.05  # 100% + 5%

# Lobe width thresholds in degrees:
#   ≥90° (1/4 of perimeter) = broad lobe (major body part)
#   ≥50° (≈1/7 of perimeter) = moderate extension
#   <50° = narrow spike
_LOBE_BROAD_DEG = 90
_LOBE_MODERATE_DEG = 50

# Polygon simplification: tolerance multiplier per iteration.
# 1.5 ≈ √(golden ratio) — moderate search step for coarse simplification.
# 1.3 ≈ 13/10 — finer step for high-res contour extraction.
_TOL_STEP_COARSE = 1.5
_TOL_STEP_FINE = 1.3

# Binary search iterations: ceil(log2(max_canvas / min_tolerance)).
# For 1000px canvas with 0.1px precision: log2(10000) ≈ 14 → 15 with margin.
_BISECT_ITERATIONS = 15
# Coarser search for grid-based simplification (integer pixel tolerance).
_SIMPLIFY_ITERATIONS_COARSE = 10
_SIMPLIFY_ITERATIONS_FINE = 12

# Binary search upper bound: half the larger canvas dimension.
# Simplification tolerance > 50% of canvas dimension produces degenerate shapes.
_BISECT_UPPER_FRACTION = 0.5

# Minimum Braille resolution: 4×4 Braille chars = 8×16 pixels,
# but we use 16×16 pixels (4×4 chars each 2×4 dots) as the floor.
_MIN_BRAILLE_RESOLUTION = 16

# Top N sub-branches to describe (limits output verbosity).
_MAX_SUB_BRANCHES = 5


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


def grid_to_halfblock(grid: NDArray[np.int8]) -> str:
    """Render grid using Unicode half-block characters for 2x vertical resolution.

    Each output character represents 2 vertical pixels using block characters:
    - \u2588 (full block) = both top and bottom filled
    - \u2580 (upper half) = top filled, bottom empty
    - \u2584 (lower half) = bottom filled, top empty
    - space = both empty

    A 32-row grid becomes 16 lines of half-block text.
    """
    rows, cols = grid.shape
    lines = []
    # Process rows in pairs
    for r in range(0, rows - 1, 2):
        line_chars = []
        for c in range(cols):
            top = grid[r, c]
            bottom = grid[r + 1, c]
            if top and bottom:
                line_chars.append("\u2588")
            elif top:
                line_chars.append("\u2580")
            elif bottom:
                line_chars.append("\u2584")
            else:
                line_chars.append(" ")
        lines.append("".join(line_chars))
    # Handle odd number of rows
    if rows % 2 == 1:
        line_chars = []
        for c in range(cols):
            line_chars.append("\u2580" if grid[rows - 1, c] else " ")
        lines.append("".join(line_chars))
    return "\n".join(lines)


def element_mini_grid(
    points: NDArray[np.float64],
    bbox: tuple[float, float, float, float],
    resolution: int = 12,
) -> str:
    """Render a single element as a small half-block grid within its bbox."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0 or len(points) == 0:
        return ""

    grid = np.zeros((resolution, resolution), dtype=np.int8)
    pts = np.asarray(points)

    cols = ((pts[:, 0] - x1) / w * resolution).astype(int)
    rows = ((pts[:, 1] - y1) / h * resolution).astype(int)
    cols = np.clip(cols, 0, resolution - 1)
    rows = np.clip(rows, 0, resolution - 1)
    grid[rows, cols] = 1

    return grid_to_halfblock(grid)


# ---------------------------------------------------------------------------
# Global silhouette descriptors
# ---------------------------------------------------------------------------

_COMPASS_8 = [
    "right", "upper-right", "up", "upper-left",
    "left", "lower-left", "down", "lower-right",
]


def _angle_to_compass(angle_deg: float) -> str:
    """Map angle in degrees (0=right, CCW) to 8-compass direction label."""
    idx = round(angle_deg / 45) % 8
    return _COMPASS_8[idx]


def _quadrant_label(r: float, c: float, rows: int, cols: int) -> str:
    """Label position within grid as a quadrant string."""
    v = "upper" if r < rows / 2 else "lower"
    h = "left" if c < cols / 2 else "right"
    if abs(r - rows / 2) < rows * _CENTER_TOLERANCE_PCT and abs(c - cols / 2) < cols * _CENTER_TOLERANCE_PCT:
        return "center"
    return f"{v}-{h}"


def skeleton_description(grid: NDArray[np.int8]) -> str:
    """Compute topological skeleton and describe branch structure.

    Uses skimage.morphology.skeletonize on the binary grid, then traces
    branches between junctions and endpoints via BFS.

    Returns natural language description like:
    "main vertical axis (18px) with 3 branches — upper-right (12px), ..."
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return ""

    from app.utils.morphology import morphological_close

    if grid is None or np.sum(grid) < _MIN_FILLED_FOR_CENTROID:
        return ""

    closed = morphological_close(grid, kernel_size=3)
    skel = skeletonize(closed.astype(bool)).astype(np.int8)

    skel_pixels = np.sum(skel)
    if skel_pixels == 0:
        return ""
    if skel_pixels <= 3:
        return "point mass (no branching)"

    rows, cols = skel.shape

    # Count 8-connected neighbors for each skeleton pixel
    neighbor_count = np.zeros_like(skel, dtype=np.int8)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(skel, -dr, axis=0), -dc, axis=1)
            # Zero out wrapped edges
            if dr == -1:
                shifted[-1, :] = 0
            elif dr == 1:
                shifted[0, :] = 0
            if dc == -1:
                shifted[:, -1] = 0
            elif dc == 1:
                shifted[:, 0] = 0
            neighbor_count += shifted

    # Classify pixels
    junctions: list[tuple[int, int]] = []
    endpoints: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if skel[r, c] == 0:
                continue
            n = neighbor_count[r, c]
            if n >= 3:
                junctions.append((r, c))
            elif n == 1:
                endpoints.append((r, c))

    # BFS-trace branches from each junction and endpoint
    visited = np.zeros_like(skel, dtype=bool)
    branches: list[list[tuple[int, int]]] = []

    start_points = junctions + endpoints
    if not start_points:
        # Simple loop with no junctions — just report total length
        return f"closed loop ({int(skel_pixels)}px), no branching"

    for sp in start_points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = sp[0] + dr, sp[1] + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and skel[nr, nc]
                    and not visited[nr, nc]
                ):
                    # Trace this branch
                    branch = [sp, (nr, nc)]
                    visited[nr, nc] = True
                    cr, cc = nr, nc
                    while True:
                        found_next = False
                        for dr2 in (-1, 0, 1):
                            for dc2 in (-1, 0, 1):
                                if dr2 == 0 and dc2 == 0:
                                    continue
                                nr2, nc2 = cr + dr2, cc + dc2
                                if (
                                    0 <= nr2 < rows
                                    and 0 <= nc2 < cols
                                    and skel[nr2, nc2]
                                    and not visited[nr2, nc2]
                                    and (nr2, nc2) not in junctions
                                ):
                                    visited[nr2, nc2] = True
                                    branch.append((nr2, nc2))
                                    cr, cc = nr2, nc2
                                    found_next = True
                                    break
                            if found_next:
                                break
                        if not found_next:
                            # Check if we hit a junction/endpoint
                            for dr2 in (-1, 0, 1):
                                for dc2 in (-1, 0, 1):
                                    if dr2 == 0 and dc2 == 0:
                                        continue
                                    nr2, nc2 = cr + dr2, cc + dc2
                                    if (
                                        0 <= nr2 < rows
                                        and 0 <= nc2 < cols
                                        and skel[nr2, nc2]
                                        and (nr2, nc2) in junctions
                                        and (nr2, nc2) != sp
                                    ):
                                        branch.append((nr2, nc2))
                                    elif (
                                        0 <= nr2 < rows
                                        and 0 <= nc2 < cols
                                        and skel[nr2, nc2]
                                        and (nr2, nc2) in endpoints
                                        and (nr2, nc2) != sp
                                    ):
                                        branch.append((nr2, nc2))
                            break
                    if len(branch) >= 2:
                        branches.append(branch)

    if not branches:
        return f"single axis ({int(skel_pixels)}px), no branching"

    # Sort by length descending
    branches.sort(key=len, reverse=True)

    # Describe main axis
    main = branches[0]
    main_len = len(main)
    sr, sc = main[0]
    er, ec = main[-1]
    dr_main = er - sr
    dc_main = ec - sc
    angle = math.degrees(math.atan2(-dr_main, dc_main)) % 360
    main_dir = _angle_to_compass(angle)

    parts = [f"main {main_dir} axis ({main_len}px)"]

    if len(branches) > 1:
        branch_descs = []
        for b in branches[1:_MAX_SUB_BRANCHES + 1]:
            b_len = len(b)
            br, bc = b[0]
            ber, bec = b[-1]
            b_dr = ber - br
            b_dc = bec - bc
            b_angle = math.degrees(math.atan2(-b_dr, b_dc)) % 360
            b_dir = _angle_to_compass(b_angle)
            b_pos = _quadrant_label(
                (br + ber) / 2, (bc + bec) / 2, rows, cols
            )
            label = "short " if b_len < main_len * _SHORT_BRANCH_RATIO else ""
            branch_descs.append(f"{b_dir} {label}({b_len}px, {b_pos})")
        parts.append(
            f"with {len(branches) - 1} branches — "
            + ", ".join(branch_descs)
        )

    parts.append(
        f"{len(junctions)} junctions, {len(endpoints)} endpoints"
    )
    return ". ".join(parts)


def radial_distance_profile(
    grid: NDArray[np.int8],
    n_rays: int = 24,
) -> tuple[str, str]:
    """Cast rays from centroid and measure distance to boundary edge.

    Returns (profile_line, features_line) describing the radial fingerprint.
    Peaks = protrusions, valleys = concavities.
    """
    from app.utils.morphology import morphological_close

    if grid is None or np.sum(grid) < _MIN_FILLED_FOR_CENTROID:
        return ("", "")

    closed = morphological_close(grid, kernel_size=3)
    filled = np.argwhere(closed > 0)
    if len(filled) == 0:
        return ("", "")

    # Centroid of filled region (row, col)
    cy, cx = filled.mean(axis=0)

    # Cast rays
    distances: list[int] = []
    rows, cols = closed.shape
    max_dim = max(rows, cols)

    for i in range(n_rays):
        angle_rad = 2 * math.pi * i / n_rays  # 0=right, CCW
        dx = math.cos(angle_rad)
        dy = -math.sin(angle_rad)  # negative because row increases downward

        # Walk outward from centroid
        max_dist = 0
        for step in range(1, max_dim + 1):
            r = int(round(cy + dy * step))
            c = int(round(cx + dx * step))
            if r < 0 or r >= rows or c < 0 or c >= cols:
                break
            if closed[r, c]:
                max_dist = step
            else:
                # Stop at first empty pixel after leaving centroid
                # (only if we've already found filled pixels)
                if max_dist > 0:
                    break
        distances.append(max_dist)

    step_deg = 360 // n_rays
    profile_line = (
        f"Radial profile (0\u00b0=right, CCW, {step_deg}\u00b0): "
        + ",".join(str(d) for d in distances)
    )

    # Find peaks and valleys via circular comparison
    d_range = max(distances) - min(distances)
    if d_range < 2:
        return (profile_line, "No significant peaks or valleys (approximately circular)")

    threshold = d_range * _PEAK_PROMINENCE_FRACTION
    peaks: list[tuple[int, int]] = []  # (angle_deg, distance)
    valleys: list[tuple[int, int]] = []

    for i in range(n_rays):
        prev_d = distances[(i - 1) % n_rays]
        curr_d = distances[i]
        next_d = distances[(i + 1) % n_rays]
        angle_deg = i * step_deg

        if curr_d > prev_d and curr_d > next_d and (curr_d - min(prev_d, next_d)) >= threshold:
            peaks.append((angle_deg, curr_d))
        if curr_d < prev_d and curr_d < next_d and (max(prev_d, next_d) - curr_d) >= threshold:
            valleys.append((angle_deg, curr_d))

    # Sort by prominence
    peaks.sort(key=lambda x: x[1], reverse=True)
    valleys.sort(key=lambda x: x[1])

    mean_dist = sum(distances) / n_rays if n_rays > 0 else 1

    # Compute angular width for each peak (base width above mean distance)
    def _peak_width(peak_idx: int, peak_dist: int) -> tuple[int, str]:
        """Compute width of the elevated region above the mean radius.

        Uses mean_dist as threshold (not half-max) to capture the full
        span of the protrusion where it rises above the body mass.
        """
        thresh = mean_dist * _PEAK_NOISE_MARGIN
        # Walk left
        left = 0
        for k in range(1, n_rays // 2 + 1):
            if distances[(peak_idx - k) % n_rays] < thresh:
                break
            left = k
        # Walk right
        right = 0
        for k in range(1, n_rays // 2 + 1):
            if distances[(peak_idx + k) % n_rays] < thresh:
                break
            right = k
        width = (left + right + 1) * step_deg
        if width >= _LOBE_BROAD_DEG:
            label = "broad lobe"
        elif width >= _LOBE_MODERATE_DEG:
            label = "moderate extension"
        else:
            label = "narrow spike"
        return (width, label)

    parts = []
    if peaks:
        peak_strs = []
        for a, d in peaks[:3]:
            idx = a // step_deg
            width_deg, width_label = _peak_width(idx, d)
            peak_strs.append(
                f"{a}\u00b0 ({d}px, {_angle_to_compass(a)}, "
                f"{width_label} ~{width_deg}\u00b0 wide)"
            )
        parts.append("Peaks: " + ", ".join(peak_strs))
    if valleys:
        valley_strs = [
            f"{a}\u00b0 ({d}px, {_angle_to_compass(a)})"
            for a, d in valleys[:3]
        ]
        parts.append("Valleys: " + ", ".join(valley_strs))

    features_line = ". ".join(parts) if parts else "No significant peaks or valleys"
    return (profile_line, features_line)


def simplified_contour_path(
    composite_silhouette: Polygon | None,
    grid: NDArray[np.int8] | None,
    canvas_w: float,
    canvas_h: float,
    max_vertices: int = 12,
) -> str:
    """Generate a simplified SVG-like outline path from the silhouette.

    Strategy A: Shapely polygon simplify (preferred).
    Strategy B: skimage find_contours on grid (fallback).
    """
    # Strategy A: Shapely polygon
    if composite_silhouette is not None:
        try:
            from shapely.geometry import Polygon as _Polygon

            if not composite_silhouette.is_empty and composite_silhouette.exterior is not None:
                coords = list(composite_silhouette.exterior.coords)
                if len(coords) > max_vertices:
                    # Binary search for tolerance
                    lo, hi = 0.0, max(canvas_w, canvas_h) * _BISECT_UPPER_FRACTION
                    best = coords
                    for _ in range(_BISECT_ITERATIONS):
                        mid = (lo + hi) / 2
                        simplified = composite_silhouette.simplify(mid)
                        if simplified.is_empty or simplified.exterior is None:
                            hi = mid
                            continue
                        sc = list(simplified.exterior.coords)
                        if len(sc) <= max_vertices:
                            best = sc
                            hi = mid
                        else:
                            lo = mid
                    coords = best
                # Remove closing duplicate
                if len(coords) > 1 and coords[0] == coords[-1]:
                    coords = coords[:-1]
                if len(coords) >= 3:
                    return _format_path(coords, len(coords))
        except Exception:
            pass

    # Strategy B: skimage find_contours
    if grid is not None and np.sum(grid) >= _MIN_FILLED_FOR_CENTROID:
        try:
            from skimage.measure import approximate_polygon, find_contours

            from app.utils.morphology import morphological_close

            closed = morphological_close(grid, kernel_size=3)
            contours = find_contours(closed.astype(float), level=0.5)
            if contours:
                # Take longest contour
                contour = max(contours, key=len)
                # Simplify
                tol = 1.0
                for _ in range(_SIMPLIFY_ITERATIONS_COARSE):
                    approx = approximate_polygon(contour, tolerance=tol)
                    if len(approx) <= max_vertices:
                        break
                    tol *= _TOL_STEP_COARSE
                # Scale to canvas coordinates
                rows, cols = grid.shape
                scaled = [
                    (c / cols * canvas_w, r / rows * canvas_h)
                    for r, c in approx
                ]
                # Remove closing duplicate
                if len(scaled) > 1 and scaled[0] == scaled[-1]:
                    scaled = scaled[:-1]
                if len(scaled) >= 3:
                    return _format_path(scaled, len(scaled))
        except Exception:
            pass

    return ""


def extract_high_res_contour(
    composite_silhouette: Polygon | None,
    grid: NDArray[np.int8] | None,
    canvas_w: float,
    canvas_h: float,
    max_vertices: int = 40,
) -> list[tuple[float, float]]:
    """Extract simplified contour as raw coordinate tuples for contour walk.

    Prefers the grid-based contour (skimage) over Shapely because for complex
    illustrations the Shapely union of many overlapping elements produces a
    near-rectangular blob, while the rasterized grid preserves the actual
    visible boundary shape.

    Returns coordinates in clockwise order.
    """

    def _ensure_clockwise(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        # Signed area (SVG y-down: positive = clockwise)
        s = 0.0
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            s += (x2 - x1) * (y2 + y1)
        if s < 0:  # CCW → reverse
            return list(reversed(pts))
        return pts

    # Strategy A (preferred): skimage grid contour — traces actual pixel boundary
    if grid is not None and np.sum(grid) >= _MIN_FILLED_FOR_CENTROID:
        try:
            from skimage.measure import approximate_polygon, find_contours

            from app.utils.morphology import morphological_close

            closed = morphological_close(grid, kernel_size=3)
            contours = find_contours(closed.astype(float), level=0.5)
            if contours:
                contour = max(contours, key=len)
                tol = 1.0
                for _ in range(_SIMPLIFY_ITERATIONS_FINE):
                    approx = approximate_polygon(contour, tolerance=tol)
                    if len(approx) <= max_vertices:
                        break
                    tol *= _TOL_STEP_FINE
                rows, cols = grid.shape
                scaled = [
                    (float(c / cols * canvas_w), float(r / rows * canvas_h))
                    for r, c in approx
                ]
                if len(scaled) > 1 and scaled[0] == scaled[-1]:
                    scaled = scaled[:-1]
                if len(scaled) >= 3:
                    return _ensure_clockwise(scaled)
        except Exception:
            pass

    # Strategy B (fallback): Shapely polygon simplification
    if composite_silhouette is not None:
        try:
            if not composite_silhouette.is_empty and composite_silhouette.exterior is not None:
                coords = list(composite_silhouette.exterior.coords)
                if len(coords) > max_vertices:
                    lo, hi = 0.0, max(canvas_w, canvas_h) * _BISECT_UPPER_FRACTION
                    best = coords
                    for _ in range(_BISECT_ITERATIONS):
                        mid = (lo + hi) / 2
                        simplified = composite_silhouette.simplify(mid)
                        if simplified.is_empty or simplified.exterior is None:
                            hi = mid
                            continue
                        sc = list(simplified.exterior.coords)
                        if len(sc) <= max_vertices:
                            best = sc
                            hi = mid
                        else:
                            lo = mid
                    coords = best
                if len(coords) > 1 and coords[0] == coords[-1]:
                    coords = coords[:-1]
                if len(coords) >= 3:
                    pts = [(float(x), float(y)) for x, y in coords]
                    return _ensure_clockwise(pts)
        except Exception:
            pass

    return []


def _format_path(coords: list, n_vertices: int) -> str:
    """Format coordinate list as SVG-like M/L path with integer coords."""
    x0, y0 = coords[0]
    parts = [f"M {int(round(x0))},{int(round(y0))}"]
    for x, y in coords[1:]:
        parts.append(f"L {int(round(x))},{int(round(y))}")
    parts.append("Z")
    return f"Simplified outline ({n_vertices} vertices): " + " ".join(parts)


# ---------------------------------------------------------------------------
# Braille rasterization — high-resolution text-art using Unicode Braille
# ---------------------------------------------------------------------------
# Each Braille character (U+2800–U+28FF) encodes a 2×4 dot matrix (8 pixels).
# A 64×64 pixel grid becomes ~32 chars wide × 16 lines = ~512 chars.
# This is 8× denser than regular text and ~4× denser than half-block.

_BRAILLE_DOT_MAP = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


def grid_to_braille(grid: NDArray[np.int8]) -> str:
    """Render a binary grid as Unicode Braille text.

    Each character represents a 2-wide × 4-tall pixel block.
    A 64×64 grid becomes 32 chars × 16 lines.
    """
    rows, cols = grid.shape
    # Pad to multiples of 4 (rows) and 2 (cols)
    pad_r = (4 - rows % 4) % 4
    pad_c = (2 - cols % 2) % 2
    if pad_r or pad_c:
        grid = np.pad(grid, ((0, pad_r), (0, pad_c)), constant_values=0)
    rows, cols = grid.shape

    lines = []
    for br in range(0, rows, 4):
        line = []
        for bc in range(0, cols, 2):
            cp = 0x2800
            for dr in range(4):
                for dc in range(2):
                    if grid[br + dr, bc + dc]:
                        cp |= _BRAILLE_DOT_MAP[dr][dc]
            line.append(chr(cp))
        lines.append("".join(line))
    return "\n".join(lines)


def _braille_resolution_from_points(n_points: int) -> int:
    """Derive Braille grid resolution from element boundary complexity.

    Each Braille character encodes a 2×4 = 8 binary pixel matrix.
    Information capacity: n_points boundary points × 1 bit = n_points bits.
    Minimum Braille characters needed: n_points / 8.
    Grid side (width in chars) = √(n_chars × 2) (correcting for 2:1
    Braille char aspect ratio: each char is 2 dots wide × 4 dots tall).
    Resolution (in pixels) = grid_side × 2 (2 pixels per Braille char width).

    Mathematical basis: information-theoretic capacity of Braille characters
    matched to the information content of the boundary.

    Must be divisible by 4 (Braille rows need 4 pixels each).
    """
    if n_points < 4:
        return _MIN_BRAILLE_RESOLUTION
    min_chars = n_points / 8  # 8 bits per Braille char
    grid_side = int(np.ceil(np.sqrt(min_chars * 2)))  # aspect correction
    resolution = grid_side * 2  # pixels per Braille char width
    resolution = max(resolution, _MIN_BRAILLE_RESOLUTION)
    # Round up to multiple of 4 (Braille row height)
    resolution = ((resolution + 3) // 4) * 4
    return resolution


def element_braille_grid(
    points: NDArray[np.float64],
    bbox: tuple[float, float, float, float],
    resolution: int | None = None,
) -> str:
    """Render a single element as a Braille grid within its bounding box.

    Args:
        points: Nx2 array of (x, y) boundary sample points.
        bbox: (x1, y1, x2, y2) bounding box.
        resolution: Grid resolution. If None, derived from len(points)
            using information-theoretic Braille capacity.

    Returns:
        Braille string or empty string if element is too small.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0 or len(points) == 0:
        return ""

    if resolution is None:
        resolution = _braille_resolution_from_points(len(points))

    grid = np.zeros((resolution, resolution), dtype=np.int8)
    pts = np.asarray(points)

    # Map points to grid coordinates within bbox, preserving aspect ratio
    dim = max(w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cols_arr = ((pts[:, 0] - (cx - dim / 2)) / dim * resolution).astype(int)
    rows_arr = ((pts[:, 1] - (cy - dim / 2)) / dim * resolution).astype(int)
    cols_arr = np.clip(cols_arr, 0, resolution - 1)
    rows_arr = np.clip(rows_arr, 0, resolution - 1)
    grid[rows_arr, cols_arr] = 1

    return grid_to_braille(grid)


def group_braille_grid(
    subpath_list: list,
    group_bbox: tuple[float, float, float, float] | None = None,
    resolution: int | None = None,
) -> str:
    """Render multiple elements combined as a single Braille grid.

    Used for concentric groups (eye, acorn, wheel) where the composite
    shape reveals what the group IS.

    Args:
        subpath_list: List of SubPathData objects to render together.
        group_bbox: Optional explicit bounding box. If None, computed from elements.
        resolution: Grid resolution. If None, derived from total point count
            using information-theoretic Braille capacity.

    Returns:
        Braille string or empty string if no renderable data.
    """
    if not subpath_list:
        return ""

    # Collect all points
    all_points = []
    for sp in subpath_list:
        pts = getattr(sp, "points", None)
        if pts is not None and len(pts) > 0:
            all_points.append(np.asarray(pts))

    if not all_points:
        return ""

    combined = np.vstack(all_points)

    # Compute bbox from elements if not provided
    if group_bbox is None:
        x1 = float(combined[:, 0].min())
        y1 = float(combined[:, 1].min())
        x2 = float(combined[:, 0].max())
        y2 = float(combined[:, 1].max())
    else:
        x1, y1, x2, y2 = group_bbox

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return ""

    if resolution is None:
        resolution = _braille_resolution_from_points(len(combined))

    grid = np.zeros((resolution, resolution), dtype=np.int8)
    dim = max(w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cols_arr = ((combined[:, 0] - (cx - dim / 2)) / dim * resolution).astype(int)
    rows_arr = ((combined[:, 1] - (cy - dim / 2)) / dim * resolution).astype(int)
    cols_arr = np.clip(cols_arr, 0, resolution - 1)
    rows_arr = np.clip(rows_arr, 0, resolution - 1)
    grid[rows_arr, cols_arr] = 1

    return grid_to_braille(grid)


def rasterize_polygon_to_grid(
    polygon: Polygon,
    canvas_w: float,
    canvas_h: float,
    resolution: int = 64,
) -> NDArray[np.int8]:
    """Rasterize a Shapely Polygon into a filled binary grid.

    Uses skimage.draw.polygon to fill the interior, producing a solid
    silhouette grid suitable for skeleton/radial analysis.

    For MultiPolygon (e.g., complex SVG paths that fragment into pieces),
    ALL geoms are rasterized to capture the complete shape.
    """
    from shapely.geometry import MultiPolygon as _MultiPolygon

    grid = np.zeros((resolution, resolution), dtype=np.int8)

    if polygon is None or polygon.is_empty:
        return grid

    # Collect all polygon components to rasterize
    geoms = []
    if isinstance(polygon, _MultiPolygon):
        geoms = list(polygon.geoms)
    else:
        geoms = [polygon]

    try:
        from skimage.draw import polygon as draw_polygon

        for geom in geoms:
            if geom.is_empty or geom.exterior is None:
                continue
            coords = list(geom.exterior.coords)
            rows_arr = [c[1] / canvas_h * resolution for c in coords]
            cols_arr = [c[0] / canvas_w * resolution for c in coords]
            rr, cc = draw_polygon(rows_arr, cols_arr, shape=(resolution, resolution))
            grid[rr, cc] = 1
    except Exception:
        # Fallback: boundary-only rasterization of all geoms
        for geom in geoms:
            if geom.is_empty or geom.exterior is None:
                continue
            coords = list(geom.exterior.coords)
            for x, y in coords:
                c = int(x / canvas_w * resolution)
                r = int(y / canvas_h * resolution)
                c = max(0, min(resolution - 1, c))
                r = max(0, min(resolution - 1, r))
                grid[r, c] = 1

    return grid


def composite_braille_grid(
    ctx_subpaths: list,
    canvas_w: float,
    canvas_h: float,
    resolution: int = 64,
) -> str:
    """Render all elements combined as a Braille silhouette.

    Args:
        ctx_subpaths: List of all SubPathData objects.
        canvas_w: Canvas width.
        canvas_h: Canvas height.
        resolution: Grid resolution (default 64 → 32 chars × 16 lines).

    Returns:
        Braille string of the composite silhouette.
    """
    all_points = []
    for sp in ctx_subpaths:
        pts = getattr(sp, "points", None)
        if pts is not None and len(pts) > 0:
            all_points.append(np.asarray(pts))

    if not all_points:
        return ""

    combined = np.vstack(all_points)
    grid = np.zeros((resolution, resolution), dtype=np.int8)

    cols_arr = ((combined[:, 0] / canvas_w) * resolution).astype(int)
    rows_arr = ((combined[:, 1] / canvas_h) * resolution).astype(int)
    cols_arr = np.clip(cols_arr, 0, resolution - 1)
    rows_arr = np.clip(rows_arr, 0, resolution - 1)
    grid[rows_arr, cols_arr] = 1

    return grid_to_braille(grid)

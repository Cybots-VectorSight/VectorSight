"""Silhouette pipeline -- polygon to smooth Bezier SVG path.

Pipeline:
  1. Even resample along arc length
  2. B-spline smoothing (scipy splprep/splev) -- smooths without deleting vertices
  3. Schneider Bezier fitting (Graphics Gems 1990) for SVG output
  4. Spike detection (Otsu on |curvature|)

Handles MultiPolygon -- returns paths for ALL parts, not just the largest.
No Douglas-Peucker -- B-spline preserves all features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu


@dataclass
class SpikeInfo:
    """A curvature spike detected on the contour."""

    pct: float
    x: float
    y: float
    magnitude: float
    sign: str  # "convex" or "concave"
    quadrant: str  # "upper-left", etc.


@dataclass
class SilhouetteResult:
    """Output of the silhouette pipeline."""

    svg_d: str = ""  # SVG path d attribute (M, C, L, Z)
    n_beziers: int = 0
    n_lines: int = 0
    total_ctrl_pts: int = 0
    spikes: list[SpikeInfo] = field(default_factory=list)
    error: str = ""


def _curvature_from_coords(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed curvature via central differences on periodic arrays."""
    dx = np.roll(x, -1) - np.roll(x, 1)
    dy = np.roll(y, -1) - np.roll(y, 1)
    ddx = np.roll(x, -1) - 2 * x + np.roll(x, 1)
    ddy = np.roll(y, -1) - 2 * y + np.roll(y, 1)
    num = dx * ddy - dy * ddx
    den = (dx**2 + dy**2) ** 1.5
    den = np.maximum(den, 1e-10)
    return num / den


def _fit_cubic_bezier_ls(points: np.ndarray):
    """Least-squares cubic Bezier fit (Schneider's method)."""
    n = len(points)
    P0, P3 = points[0], points[-1]

    if n <= 2:
        C1 = P0 + (P3 - P0) / 3
        C2 = P0 + 2 * (P3 - P0) / 3
        return C1, C2, 0.0, 0

    chords = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    total_chord = chords.sum()
    if total_chord < 1e-10:
        C1 = P0 + (P3 - P0) / 3
        C2 = P0 + 2 * (P3 - P0) / 3
        return C1, C2, 0.0, 0

    t = np.zeros(n)
    t[1:] = np.cumsum(chords) / total_chord
    t[-1] = 1.0

    A1 = 3 * (1 - t) ** 2 * t
    A2 = 3 * (1 - t) * t**2
    b0 = (1 - t) ** 3
    b3 = t**3
    d = points - b0[:, None] * P0 - b3[:, None] * P3

    m11 = np.dot(A1, A1)
    m12 = np.dot(A1, A2)
    m22 = np.dot(A2, A2)
    det = m11 * m22 - m12 * m12

    if abs(det) < 1e-10:
        C1 = P0 + (P3 - P0) / 3
        C2 = P0 + 2 * (P3 - P0) / 3
    else:
        r1 = np.array([np.dot(A1, d[:, 0]), np.dot(A1, d[:, 1])])
        r2 = np.array([np.dot(A2, d[:, 0]), np.dot(A2, d[:, 1])])
        C1 = (m22 * r1 - m12 * r2) / det
        C2 = (m11 * r2 - m12 * r1) / det

    fitted = (
        b0[:, None] * P0
        + A1[:, None] * C1
        + A2[:, None] * C2
        + b3[:, None] * P3
    )
    errors = np.sqrt(np.sum((points - fitted) ** 2, axis=1))
    return C1, C2, float(errors.max()), int(errors.argmax())


def _schneider_fit(points: np.ndarray, tol: float, depth: int = 0):
    """Recursive Schneider fit: split at worst point if error > tol."""
    max_depth = max(1, int(np.log2(max(len(points), 2))))
    if len(points) < 4 or depth > max_depth:
        if len(points) < 2:
            return []
        C1, C2, _, _ = _fit_cubic_bezier_ls(points)
        return [(points[0], C1, C2, points[-1])]

    C1, C2, max_err, worst = _fit_cubic_bezier_ls(points)
    if max_err <= tol:
        return [(points[0], C1, C2, points[-1])]

    split = max(1, min(worst, len(points) - 2))
    left = _schneider_fit(points[: split + 1], tol, depth + 1)
    right = _schneider_fit(points[split:], tol, depth + 1)
    return left + right


def _contour_to_bezier(
    coords: np.ndarray,
    canvas_w: float,
    canvas_h: float,
) -> tuple[str, int, list[SpikeInfo]]:
    """Convert a single closed contour to Bezier SVG path + spikes."""
    x, y = coords[:, 0], coords[:, 1]

    # Close if needed
    if np.allclose([x[0], y[0]], [x[-1], y[-1]], atol=0.1):
        x, y = x[:-1], y[:-1]

    if len(x) < 4:
        return "", 0, []

    # Even resample along arc length
    diffs = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arc_length = diffs.sum()
    if arc_length < 2.0:
        return "", 0, []

    n_work = max(32, min(int(arc_length), 2048))
    cumlen = np.concatenate([[0], np.cumsum(diffs)])
    even_s = np.linspace(0, cumlen[-1], n_work, endpoint=False)
    rx = np.interp(even_s, cumlen, x)
    ry = np.interp(even_s, cumlen, y)

    # B-spline smoothing
    smoothing = arc_length * 0.5
    try:
        tck, _ = splprep([rx, ry], s=smoothing, per=True, k=3)
        u_fine = np.linspace(0, 1, n_work, endpoint=False)
        sx, sy = splev(u_fine, tck)
    except (ValueError, TypeError):
        sx, sy = rx, ry

    n_pts = len(sx)

    # Schneider Bezier fit on the smoothed contour
    pts = np.column_stack([sx, sy])
    pts_closed = np.vstack([pts, pts[0]])

    work_edges = np.sqrt(np.diff(sx) ** 2 + np.diff(sy) ** 2)
    error_tol = float(np.median(work_edges))

    beziers = _schneider_fit(pts_closed, error_tol)
    if not beziers:
        return "", 0, []

    # Build SVG path
    P0 = beziers[0][0]
    path_cmds = [f"M {P0[0]:.1f},{P0[1]:.1f}"]
    for _, C1, C2, P3 in beziers:
        path_cmds.append(
            f"C {C1[0]:.1f},{C1[1]:.1f} "
            f"{C2[0]:.1f},{C2[1]:.1f} "
            f"{P3[0]:.1f},{P3[1]:.1f}"
        )
    path_cmds.append("Z")
    svg_d = " ".join(path_cmds)

    # Spike detection on smoothed contour
    kappa = _curvature_from_coords(np.asarray(sx), np.asarray(sy))
    abs_kappa = np.abs(kappa)

    try:
        spike_thresh = threshold_otsu(abs_kappa)
    except ValueError:
        spike_thresh = float(np.median(abs_kappa))

    spike_min_dist = max(3, n_pts // 20)
    spikes_idx, _ = find_peaks(
        abs_kappa, height=spike_thresh, distance=spike_min_dist
    )

    cx_mid, cy_mid = canvas_w / 2, canvas_h / 2
    spikes = []
    for sp in spikes_idx:
        x_pos, y_pos = float(sx[sp]), float(sy[sp])
        h = "left" if x_pos < cx_mid else "right"
        v = "upper" if y_pos < cy_mid else "lower"
        spikes.append(
            SpikeInfo(
                pct=sp / n_pts * 100,
                x=x_pos,
                y=y_pos,
                magnitude=float(abs_kappa[sp]),
                sign="convex" if kappa[sp] > 0 else "concave",
                quadrant=f"{v}-{h}",
            )
        )

    return svg_d, len(beziers), spikes


def _extract_polys(geom):
    """Extract all Polygon objects from any Shapely geometry."""
    polys = []
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend(g.geoms)
    return polys


def research_silhouette(
    geom,
    canvas_w: float = 256.0,
    canvas_h: float = 256.0,
) -> SilhouetteResult:
    """Run the silhouette pipeline on a Shapely geometry.

    Handles Polygon, MultiPolygon, and GeometryCollection.
    Uses B-spline smoothing (no Douglas-Peucker -- preserves all features).
    """
    result = SilhouetteResult()

    if geom.is_empty:
        result.error = "empty geometry"
        return result

    polys = _extract_polys(geom)

    if not polys:
        result.error = "no polygons"
        return result

    all_d = []
    total_beziers = 0
    all_spikes = []

    for poly in polys:
        if poly.is_empty or poly.area < 1:
            continue
        coords = np.array(poly.exterior.coords)
        if len(coords) < 4:
            continue

        svg_d, n_bez, spikes = _contour_to_bezier(coords, canvas_w, canvas_h)
        if svg_d:
            all_d.append(svg_d)
            total_beziers += n_bez
            all_spikes.extend(spikes)

    if not all_d:
        result.error = "no valid contours"
        return result

    result.svg_d = " ".join(all_d)
    result.n_beziers = total_beziers
    result.total_ctrl_pts = 1 + total_beziers * 3
    result.spikes = all_spikes
    return result

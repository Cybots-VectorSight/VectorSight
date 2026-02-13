"""Build LLM-ready enrichment text from grouped SVG features.

Pipeline: split paths -> merge overlapping -> group by proximity ->
  shape descriptors + relative positions + grids + contour walk -> enrichment text

No raw SVG paths in the prompt -- just shape descriptors the LLM can reason about.
Vocabulary is generalized (no animal-specific terms) to work for any SVG type.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from shapely.geometry import Point
from shapely.prepared import prep

from app.engine.breakdown.separate import GroupData
from app.engine.breakdown.silhouette import SilhouetteResult, _curvature_from_coords
from app.models.enrichment import (
    ElementSummary,
    EnrichmentOutput,
    SilhouetteInfo,
    SymmetryInfo,
)


# == Shape descriptors from polygon geometry ==


def _shape_descriptor(geom) -> dict | None:
    """Compute shape descriptors from a Shapely polygon."""
    if geom is None or geom.is_empty or geom.area < 1:
        return None

    hull = geom.convex_hull
    convexity = geom.area / hull.area if hull.area > 0 else 1.0

    mrr = geom.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords)
    e1 = np.array(mrr_coords[1]) - np.array(mrr_coords[0])
    e2 = np.array(mrr_coords[2]) - np.array(mrr_coords[1])
    len1, len2 = np.linalg.norm(e1), np.linalg.norm(e2)

    if len1 >= len2:
        long_edge, short_edge = len1, len2
        angle = np.degrees(np.arctan2(e1[1], e1[0]))
    else:
        long_edge, short_edge = len2, len1
        angle = np.degrees(np.arctan2(e2[1], e2[0]))

    aspect = long_edge / short_edge if short_edge > 0 else 1.0

    angle = angle % 180
    if angle > 90:
        angle -= 180

    perim = geom.length
    compactness = (4 * np.pi * geom.area) / (perim**2) if perim > 0 else 0

    if aspect < 1.5 and compactness > 0.7:
        shape = "circular"
    elif aspect < 1.5 and convexity > 0.85:
        shape = "rounded"
    elif aspect < 1.5:
        shape = "compact irregular"
    elif aspect < 3.0 and convexity > 0.8:
        shape = "oval"
    elif aspect < 3.0:
        shape = "irregular blob"
    elif convexity > 0.8:
        shape = "elongated"
    else:
        shape = "elongated irregular"

    abs_angle = abs(angle)
    if abs_angle < 15:
        orient = "horizontal"
    elif abs_angle > 75:
        orient = "vertical"
    elif angle > 0:
        orient = f"tilted {angle:.0f}\u00b0 (top-left to bottom-right)"
    else:
        orient = f"tilted {angle:.0f}\u00b0 (bottom-left to top-right)"

    return {
        "shape": shape,
        "aspect": aspect,
        "orientation": orient,
        "angle": angle,
        "convexity": convexity,
        "compactness": compactness,
    }


def _position_label(cx: float, cy: float, cw: float, ch: float) -> str:
    """Human-readable position on canvas."""
    third_h, third_v = cw / 3, ch / 3
    h = "left" if cx < third_h else ("center" if cx < 2 * third_h else "right")
    v = "top" if cy < third_v else ("middle" if cy < 2 * third_v else "bottom")
    return f"{v}-{h}"


def _position_in_parent(
    child_cx: float, child_cy: float,
    parent_bounds: tuple[float, float, float, float],
) -> str:
    """Describe position of a point relative to a parent bounding box.

    Returns e.g. "upper-left (12%, 25%)" where percentages are position
    within the parent's bounding box (0%=top/left edge, 100%=bottom/right edge).
    """
    pminx, pminy, pmaxx, pmaxy = parent_bounds
    pw, ph = pmaxx - pminx, pmaxy - pminy
    if pw < 1 or ph < 1:
        return "center"
    x_pct = (child_cx - pminx) / pw * 100
    y_pct = (child_cy - pminy) / ph * 100
    # Clamp
    x_pct = max(0, min(100, x_pct))
    y_pct = max(0, min(100, y_pct))

    v = "upper" if y_pct < 35 else ("mid" if y_pct < 65 else "lower")
    h = "left" if x_pct < 35 else ("center" if x_pct < 65 else "right")
    return f"{v}-{h} ({x_pct:.0f}%, {y_pct:.0f}%)"


def _nearest_neighbors(
    groups: list[GroupData], canvas_area: float, k: int = 3, min_area_pct: float = 0.001
) -> dict[int, list[tuple[int, float, str]]]:
    """For each significant group, find the k nearest other groups.

    Returns {gi: [(neighbor_gi, distance_px, direction_str), ...]}.
    Distance is between polygon centroids.  Direction is relative
    (e.g. "above-left", "below").
    """
    sig = [
        (gi, g) for gi, g in enumerate(groups)
        if gi > 0 and g.polygon is not None and not g.polygon.is_empty
        and g.area / canvas_area > min_area_pct
    ]
    result: dict[int, list[tuple[int, float, str]]] = {}

    for gi, g in sig:
        cx1, cy1 = g.centroid
        dists: list[tuple[int, float, str]] = []
        for gj, g2 in sig:
            if gi == gj:
                continue
            cx2, cy2 = g2.centroid
            dx, dy = cx2 - cx1, cy2 - cy1
            dist = np.sqrt(dx**2 + dy**2)

            # Direction from gi to gj
            parts = []
            if abs(dy) > 5:
                parts.append("below" if dy > 0 else "above")
            if abs(dx) > 5:
                parts.append("right" if dx > 0 else "left")
            direction = "-".join(parts) if parts else "overlapping"

            dists.append((gj, dist, direction))

        dists.sort(key=lambda x: x[1])
        result[gi] = dists[:k]

    return result



def _build_containment(groups: list[GroupData]) -> dict[int, int]:
    """Build containment map: child_gi -> parent_gi.

    A group is "inside" another if >50% of its area is contained by the other.
    The parent is the SMALLEST containing group (most specific).
    """
    n = len(groups)
    contains: dict[int, int] = {}

    for i in range(1, n):  # skip G0 (it contains everything)
        gi_poly = groups[i].polygon
        if gi_poly is None or gi_poly.is_empty:
            continue
        gi_area = groups[i].area
        if gi_area <= 0:
            continue

        best_parent = None
        best_parent_area = float("inf")

        for j in range(n):
            if i == j:
                continue
            gj_poly = groups[j].polygon
            if gj_poly is None or gj_poly.is_empty:
                continue
            gj_area = groups[j].area
            if gj_area <= gi_area:
                continue  # parent must be larger

            try:
                inter = gi_poly.intersection(gj_poly).area
                if inter / gi_area > 0.5 and gj_area < best_parent_area:
                    best_parent = j
                    best_parent_area = gj_area
            except Exception:
                continue

        if best_parent is not None:
            contains[i] = best_parent

    return contains


def _relative_position(cx1, cy1, cx2, cy2) -> str:
    """Describe position of (cx1,cy1) relative to (cx2,cy2)."""
    dx, dy = cx1 - cx2, cy1 - cy2
    dist = np.sqrt(dx**2 + dy**2)

    parts = []
    if abs(dy) > 5:
        parts.append("below" if dy > 0 else "above")
    if abs(dx) > 5:
        parts.append("right of" if dx > 0 else "left of")

    if not parts:
        return f"overlapping (dist {dist:.0f}px)"
    return f"{' and '.join(parts)} (dist {dist:.0f}px)"


# == Size tier classification ==

_LARGE_THRESHOLD = 0.10  # >10% of canvas
_MEDIUM_THRESHOLD = 0.01  # >1% of canvas


def _compute_size_tier(area: float, canvas_area: float) -> str:
    """Classify element into size tier based on area fraction."""
    if canvas_area <= 0:
        return "MEDIUM"
    ratio = area / canvas_area
    if ratio >= _LARGE_THRESHOLD:
        return "LARGE"
    elif ratio >= _MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "SMALL"


# == ASCII grid rasterization ==

_CHAR_ASPECT = 2.0  # terminal chars are ~2x taller than wide
_SILHOUETTE_LABEL = "*"
_FEATURE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _label(gi: int) -> str:
    """Get the display label for group index *gi*.

    Index 0 (silhouette) → '*', index 1+ → A, B, C, ...
    Beyond 26 features, falls back to '#27', '#28', etc.
    """
    if gi == 0:
        return _SILHOUETTE_LABEL
    fi = gi - 1  # feature index (0-based)
    if fi < len(_FEATURE_LABELS):
        return _FEATURE_LABELS[fi]
    return f"#{gi}"


def _rasterize_composite(
    groups: list[GroupData], cw: float, ch: float, grid_w: int = 48
) -> list[str]:
    """Render all groups into one labeled ASCII grid.

    G0 (largest/background) is rendered as border-only so the silhouette
    shape is visible. All other groups are rendered filled.
    """
    grid_h = max(1, round(grid_w * ch / cw / _CHAR_ASPECT))

    masks: dict[int, list[list[bool]]] = {}
    for gi, g in enumerate(groups):
        poly = g.polygon
        if poly is None or poly.is_empty or g.area < 1:
            continue
        mask = [[False] * grid_w for _ in range(grid_h)]
        prepped = prep(poly)
        for row in range(grid_h):
            cy = (row + 0.5) * ch / grid_h
            for col in range(grid_w):
                cx = (col + 0.5) * cw / grid_w
                if prepped.contains(Point(cx, cy)):
                    mask[row][col] = True
        masks[gi] = mask

    grid = [["." for _ in range(grid_w)] for _ in range(grid_h)]
    order = sorted(masks.keys(), key=lambda i: groups[i].area, reverse=True)

    for gi in order:
        mask = masks[gi]
        char = _label(gi)

        if gi == 0:
            # Border-only for silhouette (A)
            for row in range(grid_h):
                for col in range(grid_w):
                    if not mask[row][col]:
                        continue
                    is_edge = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if (
                            nr < 0
                            or nr >= grid_h
                            or nc < 0
                            or nc >= grid_w
                            or not mask[nr][nc]
                        ):
                            is_edge = True
                            break
                    if is_edge:
                        grid[row][col] = char
        else:
            for row in range(grid_h):
                for col in range(grid_w):
                    if mask[row][col]:
                        grid[row][col] = char

    return ["".join(r) for r in grid]


def _rasterize_braille(
    polygon, cw: float, ch: float, char_w: int = 60, border_only: bool = False
) -> list[str]:
    """Render a polygon as Unicode Braille silhouette.

    Each Braille char encodes a 2x4 pixel block -> 8x resolution vs ASCII.
    """
    px_w = char_w * 2
    px_h_raw = round(px_w * ch / cw)
    px_h = ((px_h_raw + 3) // 4) * 4
    char_h = px_h // 4

    prepped = prep(polygon)
    bitmap = [[False] * px_w for _ in range(px_h)]
    for py in range(px_h):
        cy = (py + 0.5) * ch / px_h
        for px in range(px_w):
            cx = (px + 0.5) * cw / px_w
            if prepped.contains(Point(cx, cy)):
                bitmap[py][px] = True

    if border_only:
        border = [[False] * px_w for _ in range(px_h)]
        for py in range(px_h):
            for px in range(px_w):
                if not bitmap[py][px]:
                    continue
                is_edge = False
                for dy, dx in [
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1),
                ]:
                    ny, nx = py + dy, px + dx
                    if (
                        ny < 0
                        or ny >= px_h
                        or nx < 0
                        or nx >= px_w
                        or not bitmap[ny][nx]
                    ):
                        is_edge = True
                        break
                if is_edge:
                    border[py][px] = True
        bitmap = border

    _BRAILLE_BASE = 0x2800
    _DOT_MAP = [
        (0, 0, 0x01), (1, 0, 0x02), (2, 0, 0x04),
        (0, 1, 0x08), (1, 1, 0x10), (2, 1, 0x20),
        (3, 0, 0x40), (3, 1, 0x80),
    ]

    lines = []
    for cr in range(char_h):
        chars = []
        for cc in range(char_w):
            code = 0
            for dy, dx, bit in _DOT_MAP:
                py = cr * 4 + dy
                px = cc * 2 + dx
                if py < px_h and px < px_w and bitmap[py][px]:
                    code |= bit
            chars.append(chr(_BRAILLE_BASE + code))
        lines.append("".join(chars))

    return lines


def _rasterize_group_cropped(
    polygon, bounds: tuple, char: str, max_w: int = 40, border_only: bool = False
) -> list[str]:
    """Render a single group as an ASCII grid cropped to its bounding box.

    The grid is scaled so the longest axis fits in *max_w* characters.
    With border_only=True, only edge cells are drawn so the outline shape
    is visible (solid fills are just blobs, unhelpful for shape recognition).
    """
    minx, miny, maxx, maxy = bounds
    bw, bh = maxx - minx, maxy - miny

    # Scale so the wider dimension fits max_w, respecting char aspect ratio
    if bw >= bh:
        grid_w = min(max_w, max(8, round(bw / max(bw, bh) * max_w)))
        grid_h = max(1, round(grid_w * bh / bw / _CHAR_ASPECT))
    else:
        grid_h = min(max_w, max(4, round(bh / max(bw, bh) * max_w / _CHAR_ASPECT)))
        grid_w = max(1, round(grid_h * bw / bh * _CHAR_ASPECT))

    grid_w = max(4, min(grid_w, max_w))
    grid_h = max(2, min(grid_h, 30))

    prepped = prep(polygon)
    # Build fill mask first
    mask = [[False] * grid_w for _ in range(grid_h)]
    for row in range(grid_h):
        cy = miny + (row + 0.5) * bh / grid_h
        for col in range(grid_w):
            cx = minx + (col + 0.5) * bw / grid_w
            if prepped.contains(Point(cx, cy)):
                mask[row][col] = True

    rows = []
    for row in range(grid_h):
        line = []
        for col in range(grid_w):
            if not mask[row][col]:
                line.append(".")
            elif not border_only:
                line.append(char)
            else:
                # Border-only: draw char only if adjacent to an empty cell
                is_edge = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if nr < 0 or nr >= grid_h or nc < 0 or nc >= grid_w or not mask[nr][nc]:
                        is_edge = True
                        break
                line.append(char if is_edge else ".")
        rows.append("".join(line))
    return rows


# == Protrusion detection ==


def _detect_protrusions(
    polygon, cw: float, ch: float, n_samples: int = 72
) -> list[dict]:
    """Detect significant protrusions from the silhouette outline."""
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    coords = np.array(polygon.exterior.coords)
    x, y = coords[:, 0], coords[:, 1]
    if np.allclose([x[0], y[0]], [x[-1], y[-1]], atol=0.1):
        x, y = x[:-1], y[:-1]
    if len(x) < 10:
        return []

    cx_c, cy_c = polygon.centroid.x, polygon.centroid.y

    diffs = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arc_len = diffs.sum()
    if arc_len < 2:
        return []
    cumlen = np.concatenate([[0], np.cumsum(diffs)])
    even_s = np.linspace(0, cumlen[-1], n_samples, endpoint=False)
    sx = np.interp(even_s, cumlen, x)
    sy = np.interp(even_s, cumlen, y)

    radii = np.sqrt((sx - cx_c) ** 2 + (sy - cy_c) ** 2)
    angles = np.degrees(np.arctan2(-(sy - cy_c), sx - cx_c))

    median_r = np.median(radii)
    peaks, _ = find_peaks(
        radii,
        height=median_r * 1.05,
        distance=n_samples // 12,
        prominence=median_r * 0.05,
    )

    protrusions = []
    for pk in peaks:
        angle = angles[pk]
        if angle < 0:
            angle += 360
        dirs = [
            "right", "upper-right", "up", "upper-left",
            "left", "lower-left", "down", "lower-right",
        ]
        dir_idx = int((angle + 22.5) / 45) % 8
        direction = dirs[dir_idx]

        h = (
            "left"
            if sx[pk] < cw * 0.33
            else ("center" if sx[pk] < cw * 0.67 else "right")
        )
        v = (
            "top"
            if sy[pk] < ch * 0.33
            else ("middle" if sy[pk] < ch * 0.67 else "bottom")
        )
        pos = f"{v}-{h}"

        extension = (radii[pk] - median_r) / median_r * 100

        protrusions.append(
            {
                "angle": angle,
                "direction": direction,
                "position": pos,
                "extension_pct": extension,
                "x": float(sx[pk]),
                "y": float(sy[pk]),
            }
        )

    protrusions.sort(key=lambda p: p["extension_pct"], reverse=True)
    return protrusions


# == Generalized feature role inference ==


def _infer_feature_role(
    group: GroupData, gi: int, cw: float, ch: float
) -> str:
    """Infer the visual role of a group based on position, size, and shape.

    Uses generic spatial vocabulary (no animal-specific terms).
    """
    cx_g, cy_g = group.centroid
    area_pct = group.area / (cw * ch) * 100
    b = group.polygon.bounds
    w, h = b[2] - b[0], b[3] - b[1]
    aspect = max(w, h) / max(min(w, h), 1)

    sd = _shape_descriptor(group.polygon)
    if not sd:
        return "detail"

    y_norm = cy_g / ch  # 0=top, 1=bottom
    x_norm = cx_g / cw  # 0=left, 1=right

    if gi == 0:
        return "overall silhouette"

    # Large features (>10% of canvas)
    if area_pct > 10:
        if y_norm < 0.4:
            return "primary upper region"
        elif y_norm < 0.7:
            return "central region"
        else:
            return "lower region"

    # Medium features (2-10%)
    if area_pct > 2:
        if y_norm < 0.35:
            return "upper feature"
        elif aspect > 2.5:
            return "elongated extension"
        elif y_norm > 0.65:
            return "lower extension"
        else:
            return "central feature"

    # Small-medium features (0.5-2%)
    if area_pct > 0.5:
        if sd["compactness"] > 0.55:
            if y_norm < 0.5:
                return "small circular upper feature"
            else:
                return "small circular feature"
        elif aspect > 3.0:
            if y_norm < 0.25:
                return "protruding upper feature"
            elif y_norm > 0.7:
                return "lower narrow feature"
            else:
                return "linear feature"
        elif y_norm > 0.7:
            return "lower detail"
        else:
            return "interior feature"

    # Tiny features (<0.5%)
    if sd["compactness"] > 0.55:
        if y_norm < 0.5:
            return "small dot/circle"
        return "small accent"
    if aspect > 3.0 and y_norm < 0.25:
        return "upper narrow feature"
    if y_norm > 0.7:
        return "lower detail"
    return "detail"


# == Symmetric pair detection ==


def _find_symmetric_pairs(
    groups: list[GroupData], cw: float, ch: float, area_thresh: float = 0.001
) -> list[tuple[int, int]]:
    """Find groups that appear to be symmetric pairs."""
    canvas_area = cw * ch
    sig = [
        (gi, g)
        for gi, g in enumerate(groups)
        if gi > 0 and g.area / canvas_area > area_thresh
    ]

    pairs = []
    used: set[int] = set()
    for i, (gi, g1) in enumerate(sig):
        if gi in used:
            continue
        for gj, g2 in sig[i + 1 :]:
            if gj in used:
                continue
            a1, a2 = g1.area, g2.area
            if min(a1, a2) / max(a1, a2) < 0.5:
                continue
            cy1, cy2 = g1.centroid[1], g2.centroid[1]
            if abs(cy1 - cy2) > ch * 0.1:
                continue
            cx1, cx2 = g1.centroid[0], g2.centroid[0]
            if abs(cx1 - cx2) < cw * 0.05:
                continue
            sd1 = _shape_descriptor(g1.polygon)
            sd2 = _shape_descriptor(g2.polygon)
            if sd1 and sd2 and sd1["shape"] == sd2["shape"]:
                pairs.append((gi, gj))
                used.add(gi)
                used.add(gj)
                break

    return pairs


# == Symmetry analysis ==


def _detect_axis_type(
    sym_pairs: list[tuple[int, int]],
    groups: list[GroupData],
    cw: float,
) -> str:
    """Detect symmetry axis type from symmetric pairs."""
    if not sym_pairs:
        return "none"

    # Check if pairs are roughly mirrored across vertical center
    vertical_count = 0
    for gi, gj in sym_pairs:
        cx1 = groups[gi].centroid[0]
        cx2 = groups[gj].centroid[0]
        mid = (cx1 + cx2) / 2
        if abs(mid - cw / 2) < cw * 0.15:
            vertical_count += 1

    if vertical_count > len(sym_pairs) // 2:
        return "vertical"
    return "approximate"


def _symmetry_score(
    sym_pairs: list[tuple[int, int]], groups: list[GroupData]
) -> float:
    """Compute an overall symmetry score (0-1)."""
    if not sym_pairs or len(groups) <= 1:
        return 0.0
    # Fraction of non-background groups involved in pairs
    n_paired = len(sym_pairs) * 2
    n_non_bg = max(1, len(groups) - 1)
    return min(1.0, n_paired / n_non_bg)


# == Build enrichment text (for LLM injection) ==


def _shape_trace(polygon, cw: float, ch: float) -> str:
    """Compact directional trace of a shape's outline.

    Format:  {arrow}{degree}° {type}{percent}% → ... ⟲
    - Arrow: quick visual (↑↗→↘↓↙←↖)
    - Degree: compass heading rounded to 15° (0°=up, 90°=right, 180°=down, 270°=left)
    - Type: ~ = curve, ─ = straight (single char)
    - Percent: segment length as % of total perimeter

    Simple shapes:  ○ ~20px radius | △ pointing up, 60x80px | □ 40x80px
    Complex shapes: ↗45° ~30% → ↘135° ─15% → ↙225° ~25% → ↖315° ─10% ⟲
    """
    import math as _math

    if polygon is None or polygon.is_empty:
        return ""
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)
    if polygon.geom_type != "Polygon":
        return ""

    bounds = polygon.bounds
    bw = bounds[2] - bounds[0]
    bh = bounds[3] - bounds[1]
    if bw < 1 or bh < 1:
        return ""

    sd = _shape_descriptor(polygon)
    if not sd:
        return ""

    # Simple shape shortcuts
    if sd["compactness"] > 0.75 and sd["aspect"] < 1.5:
        radius = _math.sqrt(polygon.area / _math.pi)
        return f"\u25cb ~{radius:.0f}px radius"

    # Get raw exterior coords and simplify to key vertices
    orig_coords = list(polygon.exterior.coords)
    n_orig = len(orig_coords)
    perimeter = polygon.length
    if perimeter < 1:
        return ""

    # Adaptive Douglas-Peucker: aim for 6-12 key points
    tolerance = perimeter * 0.03
    for _ in range(5):
        simplified = polygon.simplify(tolerance)
        if simplified.is_empty or simplified.geom_type != "Polygon":
            break
        key_pts = list(simplified.exterior.coords)
        if key_pts and key_pts[0] == key_pts[-1]:
            key_pts = key_pts[:-1]
        if len(key_pts) <= 12:
            break
        tolerance *= 1.5
    else:
        key_pts = list(polygon.simplify(tolerance).exterior.coords)
        if key_pts and key_pts[0] == key_pts[-1]:
            key_pts = key_pts[:-1]

    n_pts = len(key_pts)
    if n_pts < 3:
        return ""

    # Simple polygon shortcuts
    if n_pts == 3:
        cy_avg = sum(p[1] for p in key_pts) / 3
        top = min(key_pts, key=lambda p: p[1])
        if top[1] < cy_avg - bh * 0.15:
            return f"\u25b3 pointing up, {bw:.0f}x{bh:.0f}px"
        bottom = max(key_pts, key=lambda p: p[1])
        if bottom[1] > cy_avg + bh * 0.15:
            return f"\u25bd pointing down, {bw:.0f}x{bh:.0f}px"
        return f"\u25b3 {bw:.0f}x{bh:.0f}px"

    if n_pts == 4 and sd["convexity"] > 0.85:
        return f"\u25a1 {bw:.0f}x{bh:.0f}px, {sd['orientation']}"

    # Determine if the outline is curvy vs angular
    # Many original points between few key points → curves
    is_curvy = n_orig > n_pts * 4
    stroke_char = "~" if is_curvy else "\u2500"

    # Compute total perimeter of simplified polygon
    total_len = sum(
        _math.sqrt((key_pts[(i + 1) % n_pts][0] - key_pts[i][0]) ** 2 +
                   (key_pts[(i + 1) % n_pts][1] - key_pts[i][1]) ** 2)
        for i in range(n_pts)
    )
    if total_len < 1:
        return ""

    # Build segments: (compass_degree, percent, raw_len)
    segments: list[tuple[int, int, float]] = []
    for i in range(n_pts):
        x1, y1 = key_pts[i]
        x2, y2 = key_pts[(i + 1) % n_pts]
        dx, dy = x2 - x1, y2 - y1
        seg_len = _math.sqrt(dx ** 2 + dy ** 2)

        # Compass heading: 0°=up, 90°=right, 180°=down, 270°=left
        # In SVG, y increases downward, so dy>0 means going down (180°)
        math_angle = _math.degrees(_math.atan2(dx, -dy))  # atan2(east, north)
        if math_angle < 0:
            math_angle += 360
        # Round to nearest 15°
        compass = round(math_angle / 15) * 15
        if compass == 360:
            compass = 0

        pct = round(seg_len / total_len * 100)
        segments.append((compass, pct, seg_len))

    # Merge consecutive segments with same compass heading
    merged: list[tuple[int, int]] = []  # (compass, total_pct)
    for compass, pct, _ in segments:
        if merged and merged[-1][0] == compass:
            merged[-1] = (compass, merged[-1][1] + pct)
        else:
            merged.append((compass, pct))

    # Arrow labels for quick visual scanning
    _ARROWS = [
        "\u2191",   # 0° = up
        "\u2197",   # 45° = upper-right
        "\u2192",   # 90° = right
        "\u2198",   # 135° = lower-right
        "\u2193",   # 180° = down
        "\u2199",   # 225° = lower-left
        "\u2190",   # 270° = left
        "\u2196",   # 315° = upper-left
    ]

    def _arrow(deg: int) -> str:
        idx = round(deg / 45) % 8
        return _ARROWS[idx]

    # Build trace: "↗45° ~25% → ↘135° ─15% → ..."
    parts = [f"{_arrow(deg)}{deg}\u00b0 {stroke_char}{pct}%" for deg, pct in merged]
    return " \u2192 ".join(parts) + " \u27f2"


def build_enrichment_text(
    groups: list[GroupData],
    silhouettes: list[SilhouetteResult | None],
    cw: float,
    ch: float,
) -> str:
    """Build the complete enrichment prompt text from breakdown data.

    Structure (designed for identification — perceptual hierarchy):
      1. VoT reasoning instructions
      2. Overall shape (G0 silhouette, protrusions, orientation)
      3. Group anatomy — every group with ASCII grid + simplified path + descriptors,
         ordered largest-to-smallest (biggest structures define identity,
         smaller ones confirm/refine)
      4. Composite feature map (all groups on one grid)
      5. Key observations summary
    """
    canvas_area = cw * ch
    _MIN_AREA_PCT = 0.001  # minimum to appear in enrichment at all

    # Pre-compute symmetric pairs so we can annotate inline
    sym_pairs = _find_symmetric_pairs(groups, cw, ch)
    sym_map: dict[int, int] = {}  # gi -> its mirror partner
    for a, b in sym_pairs:
        sym_map[a] = b
        sym_map[b] = a

    # Pre-compute containment hierarchy
    containment = _build_containment(groups)
    # Invert: parent -> list of children
    children_of: dict[int, list[int]] = {}
    for child, par in containment.items():
        children_of.setdefault(par, []).append(child)

    # Pre-compute nearest neighbors
    neighbors = _nearest_neighbors(groups, canvas_area, k=3, min_area_pct=_MIN_AREA_PCT)

    P: list[str] = []

    # ── VoT reasoning instructions ──────────────────────────────────
    P.append("You are identifying what an SVG icon depicts from its vector geometry.")
    P.append("")
    P.append("**How to read this data:**")
    P.append("- The SILHOUETTE shows the overall outline — look for the general form.")
    P.append(
        "- PROTRUSIONS are bumps/extensions from the body — match them to "
        "appendages, limbs, tails, antennae, handles, etc."
    )
    P.append(
        "- GROUP ANATOMY lists every interior feature from **largest to smallest**, "
        "labeled A-Z (silhouette is `*`). The biggest pieces define WHAT it is; "
        "smaller ones confirm. Each group has: an ASCII outline grid "
        "(letter = border, `.` = empty), a directional trace showing how the "
        "outline flows (arrows + straight/curved), "
        "shape/position descriptors, and containment (what's inside what)."
    )
    P.append(
        "- CONTAINMENT shows which groups are INSIDE others — "
        "a group inside another is a sub-feature of it (e.g., pupil inside eye, "
        "button on torso). Position is given as % within the parent's bounds."
    )
    P.append(
        "- SYMMETRIC PAIRS are noted inline — mirrored features strongly suggest "
        "eyes, ears, wings, wheels, handles, etc."
    )
    P.append(
        "- The COMPOSITE MAP shows how all groups relate spatially on one grid."
    )
    P.append("")
    P.append("**Reasoning steps:**")
    P.append(
        "1. Study the silhouette outline — what overall form do you see? "
        "Upright figure, horizontal animal, round object, abstract symbol?"
    )
    P.append(
        "2. Match protrusions to body parts or structural features."
    )
    P.append(
        "3. Walk through the group anatomy (largest first) — each group's "
        "ASCII grid shows what that piece looks like in isolation. "
        "Its position tells you WHERE on the subject it sits."
    )
    P.append(
        "4. Symmetric pairs are strong clues — what comes in matched pairs?"
    )
    P.append(
        "5. Give your **top 3 guesses** ranked by confidence. "
        "For each guess, cite which groups/protrusions support it."
    )
    P.append("")

    # ── Canvas info ──────────────────────────────────────────────────
    P.append(
        f"Canvas: {cw:.0f} x {ch:.0f}px | {len(groups)} groups"
    )
    P.append("")

    # ── Containment tree (quick hierarchy overview) ──────────────────
    if containment:
        P.append("## Containment Hierarchy")
        P.append("Which groups are inside which (sub-features of their parent):")

        def _tree_line(parent_gi: int, depth: int) -> list[str]:
            lines = []
            kids = children_of.get(parent_gi, [])
            kids_sorted = sorted(kids, key=lambda k: groups[k].area, reverse=True)
            for k in kids_sorted:
                indent = "  " * (depth + 1)
                role = _infer_feature_role(groups[k], k, cw, ch)
                lines.append(f"{indent}└─ {_label(k)} — {role}")
                lines.extend(_tree_line(k, depth + 1))
            return lines

        P.append(f"{_label(0)} — overall silhouette")
        # Show root-level groups (not contained by anything except silhouette)
        roots = [
            gi for gi, g in enumerate(groups)
            if gi > 0
            and g.polygon is not None
            and not g.polygon.is_empty
            and g.area / canvas_area > _MIN_AREA_PCT
            and gi not in containment
        ]
        roots.sort(key=lambda k: groups[k].area, reverse=True)
        for r in roots:
            role = _infer_feature_role(groups[r], r, cw, ch)
            P.append(f"  └─ {_label(r)} — {role}")
            P.extend(_tree_line(r, 1))

        # Groups contained directly by silhouette (parent=0 in containment map)
        g0_kids = [gi for gi, par in containment.items() if par == 0]
        g0_kids.sort(key=lambda k: groups[k].area, reverse=True)
        for k in g0_kids:
            role = _infer_feature_role(groups[k], k, cw, ch)
            P.append(f"  └─ {_label(k)} — {role} [inside {_label(0)}]")
            P.extend(_tree_line(k, 1))

        P.append("")

    # ── Overall Shape (G0) ───────────────────────────────────────────
    if not groups:
        return "\n".join(P)

    g0_poly = groups[0].polygon
    g0_hull = g0_poly.convex_hull
    g0_convexity = g0_poly.area / g0_hull.area if g0_hull.area > 0 else 1.0
    g0_sd = _shape_descriptor(g0_poly)
    g0_bounds = g0_poly.bounds
    g0_w = g0_bounds[2] - g0_bounds[0]
    g0_h = g0_bounds[3] - g0_bounds[1]

    P.append(f"## Overall Shape ({_label(0)} — outer silhouette)")
    P.append(
        f"- Bounding box: {g0_w:.0f} x {g0_h:.0f}px "
        f"(aspect {g0_w / g0_h:.1f}:1, "
        f"{'wider' if g0_w > g0_h else 'taller'} than wide)"
    )
    if g0_sd:
        P.append(f"- Shape: **{g0_sd['shape']}**, {g0_sd['orientation']}")
    P.append(
        f"- Convexity: {g0_convexity:.0%} — "
        f"{'smooth/rounded' if g0_convexity > 0.85 else 'moderately irregular' if g0_convexity > 0.7 else 'very irregular with protrusions/concavities'}"
    )

    # Orientation from minimum rotated rectangle
    g0_mrr = g0_hull.minimum_rotated_rectangle
    g0_mrr_coords = list(g0_mrr.exterior.coords)
    g0_e1 = np.array(g0_mrr_coords[1]) - np.array(g0_mrr_coords[0])
    g0_e2 = np.array(g0_mrr_coords[2]) - np.array(g0_mrr_coords[1])
    g0_len1, g0_len2 = np.linalg.norm(g0_e1), np.linalg.norm(g0_e2)
    g0_long = max(g0_len1, g0_len2)
    g0_short = min(g0_len1, g0_len2)
    g0_true_aspect = g0_long / g0_short if g0_short > 0 else 1.0
    g0_angle = np.degrees(
        np.arctan2(g0_e1[1], g0_e1[0])
        if g0_len1 >= g0_len2
        else np.arctan2(g0_e2[1], g0_e2[0])
    )
    g0_angle = g0_angle % 180
    if g0_angle > 90:
        g0_angle -= 180

    if g0_true_aspect > 1.3 and abs(g0_angle) > 60:
        P.append(
            f"- Orientation: **upright/vertical** (aspect {g0_true_aspect:.1f}:1)"
        )
    elif g0_true_aspect > 1.3 and abs(g0_angle) < 30:
        P.append(
            f"- Orientation: **horizontal/landscape** (aspect {g0_true_aspect:.1f}:1)"
        )
    elif g0_true_aspect > 1.3:
        P.append(
            f"- Orientation: tilted {g0_angle:.0f}\u00b0 (aspect {g0_true_aspect:.1f}:1)"
        )
    else:
        P.append(f"- Orientation: compact/square (aspect {g0_true_aspect:.1f}:1)")

    # Center of mass (semantic only — raw coords are not useful to LLMs)
    sig_groups = [
        (gi, g) for gi, g in enumerate(groups) if g.area / canvas_area > _MIN_AREA_PCT
    ]
    _total_area = sum(g.area for _, g in sig_groups)
    if _total_area > 0:
        _cx_mass = sum(g.centroid[0] * g.area for _, g in sig_groups) / _total_area
        _cy_mass = sum(g.centroid[1] * g.area for _, g in sig_groups) / _total_area
        mass_h = "centered" if abs(_cx_mass - cw / 2) < cw * 0.1 else (
            "shifted left" if _cx_mass < cw / 2 else "shifted right"
        )
        mass_v = "centered" if abs(_cy_mass - ch / 2) < ch * 0.1 else (
            "shifted up" if _cy_mass < ch / 2 else "shifted down"
        )
        P.append(f"- Weight distribution: {mass_h}, {mass_v}")

    # Contour description for G0 — from polygon geometry, not raw path
    protrusions_g0 = _detect_protrusions(g0_poly, cw, ch)
    n_prot = len(protrusions_g0)
    if n_prot == 0:
        P.append("- Contour: **smooth**, no significant protrusions")
    elif n_prot <= 2:
        dirs = [p["direction"] for p in protrusions_g0[:2]]
        P.append(f"- Contour: {n_prot} protrusion(s) extending **{', '.join(dirs)}**")
    else:
        dirs = [p["direction"] for p in protrusions_g0[:4]]
        P.append(f"- Contour: **complex** with {n_prot} protrusions ({', '.join(dirs)})")

    # Shape trace for silhouette
    g0_trace = _shape_trace(g0_poly, cw, ch)
    if g0_trace:
        P.append(f"- Trace: {g0_trace}")

    if sym_pairs:
        P.append(f"- **{len(sym_pairs)} symmetric pair(s)** detected")
    P.append("")

    # Silhouette as ASCII border-only (outline shape visible, not a solid blob)
    if groups[0].polygon is not None and not groups[0].polygon.is_empty:
        sil_grid = _rasterize_group_cropped(
            groups[0].polygon, g0_bounds, "#", max_w=50, border_only=True
        )
        P.append("### Silhouette (ASCII — `#` = outline, `.` = empty)")
        P.append("```")
        for line in sil_grid:
            P.append(line)
        P.append("```")
        P.append("")

    # Protrusions
    protrusions = _detect_protrusions(groups[0].polygon, cw, ch)
    if protrusions:
        P.append("### Protrusions")
        for i, pr in enumerate(protrusions[:8]):
            P.append(
                f"  {i + 1}. **{pr['direction']}** at ({pr['x']:.0f},{pr['y']:.0f}) "
                f"[{pr['position']}] — {pr['extension_pct']:.0f}% beyond avg radius"
            )
        P.append("")

    # ── Group Anatomy (largest → smallest) ───────────────────────────
    # Collect all groups with valid polygons, skip G0 (already covered above)
    anatomy_groups = [
        (gi, g)
        for gi, g in enumerate(groups)
        if gi > 0
        and g.polygon is not None
        and not g.polygon.is_empty
        and g.area / canvas_area > _MIN_AREA_PCT
    ]
    # Sort by area descending — biggest structures first
    anatomy_groups.sort(key=lambda x: x[1].area, reverse=True)

    if anatomy_groups:
        n_anat = len(anatomy_groups)
        P.append(f"## Group Anatomy ({n_anat} features, largest → smallest)")
        P.append(
            "Each group shows: role, position, shape descriptor, "
            "ASCII outline, and directional trace."
        )
        P.append("")

        for rank, (gi, g) in enumerate(anatomy_groups, 1):
            lbl = _label(gi)
            role = _infer_feature_role(g, gi, cw, ch)
            area_pct = g.area / canvas_area * 100
            b = g.polygon.bounds
            bw, bh = b[2] - b[0], b[3] - b[1]
            sd = _shape_descriptor(g.polygon)

            # Header line
            P.append(f"### #{rank} {lbl} — {role}")

            # Descriptors
            desc = f"  {area_pct:.1f}% of canvas, {bw:.0f}x{bh:.0f}px"
            if sd:
                desc += f", **{sd['shape']}** {sd['orientation']}"
            P.append(desc)

            # Containment + position relative to parent
            parent_gi = containment.get(gi)
            if parent_gi is not None:
                parent_bounds = groups[parent_gi].polygon.bounds
                rel_pos = _position_in_parent(
                    g.centroid[0], g.centroid[1], parent_bounds
                )
                P.append(f"  **inside {_label(parent_gi)}** at {rel_pos}")
            else:
                rel_pos = _position_in_parent(
                    g.centroid[0], g.centroid[1], g0_bounds
                )
                P.append(f"  position in silhouette: {rel_pos}")

            # Children (what's inside this group)
            kids = children_of.get(gi, [])
            if kids:
                P.append(f"  contains: {', '.join(_label(k) for k in kids)}")

            # Symmetric pair annotation
            if gi in sym_map:
                partner = sym_map[gi]
                par_gi = containment.get(partner)
                par_bounds = groups[par_gi].polygon.bounds if par_gi is not None else groups[0].polygon.bounds
                p_rel = _position_in_parent(
                    groups[partner].centroid[0], groups[partner].centroid[1],
                    par_bounds,
                )
                P.append(f"  **\u2194 Mirror pair with {_label(partner)} at {p_rel}**")

            # ASCII grid for every group — border-only so outline shape is visible
            # (solid fill is just a blob of letters, useless for shape recognition)
            if bw >= 1 and bh >= 1:
                max_grid = 40 if area_pct > 1.0 else (25 if area_pct > 0.3 else 12)
                grid_lines = _rasterize_group_cropped(
                    g.polygon, b, lbl, max_w=max_grid, border_only=True
                )
                P.append("```")
                for line in grid_lines:
                    P.append(line)
                P.append("```")

            # Shape trace — directional outline
            trace = _shape_trace(g.polygon, cw, ch)
            if trace:
                P.append(f"  trace: {trace}")

            # Nearest neighbors
            nn = neighbors.get(gi, [])
            if nn:
                nn_parts = [f"{_label(ngi)} {ndist:.0f}px {ndir}" for ngi, ndist, ndir in nn]
                P.append(f"  neighbors: {', '.join(nn_parts)}")

            P.append("")

    # ── Composite feature map ────────────────────────────────────────
    composite_grid = _rasterize_composite(groups, cw, ch, grid_w=60)
    P.append("## Composite Map (all groups on one grid)")
    P.append(
        f"{_label(0)} = silhouette border, A-Z = interior features by size. `.` = empty."
    )
    P.append("```")
    for line in composite_grid:
        P.append(line)
    P.append("```")
    P.append("")

    # ── Key observations summary ─────────────────────────────────────
    P.append("## Summary")
    obs: list[str] = []

    if protrusions:
        biggest = protrusions[0]
        obs.append(
            f"Largest protrusion: **{biggest['direction']}** "
            f"from {biggest['position']}, {biggest['extension_pct']:.0f}% beyond avg"
        )

    if sym_pairs:
        pair_strs = [f"{_label(a)}+{_label(b)}" for a, b in sym_pairs]
        obs.append(f"Symmetric pairs: {', '.join(pair_strs)}")

    # Detail clustering
    _small_groups = [
        (gi, g) for gi, g in sig_groups if g.area / canvas_area < 0.01
    ]
    if _small_groups:
        _small_cx = np.mean([g.centroid[0] for _, g in _small_groups])
        _detail_side = "right" if _small_cx > cw / 2 else "left"
        obs.append(f"Small details cluster on the **{_detail_side}** side")

    n_large = sum(1 for _, g in anatomy_groups if g.area / canvas_area > 0.10)
    n_med = sum(
        1
        for _, g in anatomy_groups
        if 0.01 < g.area / canvas_area <= 0.10
    )
    n_small = len(anatomy_groups) - n_large - n_med
    obs.append(f"Feature tiers: {n_large} large, {n_med} medium, {n_small} small")

    for o in obs:
        P.append(f"- {o}")

    P.append("")
    P.append(
        "Now identify the subject. Give your **top 3 guesses** ranked by confidence. "
        "For each, cite specific groups (by letter) and protrusions that support it."
    )

    return "\n".join(P)


# == Build EnrichmentOutput (for API response) ==


def build_enrichment_output(
    groups: list[GroupData],
    silhouettes: list[SilhouetteResult | None],
    cw: float,
    ch: float,
    n_raw: int,
) -> EnrichmentOutput:
    """Build the full EnrichmentOutput Pydantic model from breakdown data."""
    canvas_area = cw * ch

    elements = []
    for gi, g in enumerate(groups):
        sd = _shape_descriptor(g.polygon)
        elements.append(
            ElementSummary(
                id=_label(gi),
                shape_class=sd["shape"] if sd else "organic",
                area=g.area,
                bbox=g.polygon.bounds if g.polygon else (0, 0, 0, 0),
                centroid=g.centroid,
                circularity=sd["compactness"] if sd else 0.0,
                convexity=sd["convexity"] if sd else 0.0,
                aspect_ratio=sd["aspect"] if sd else 1.0,
                size_tier=_compute_size_tier(g.area, canvas_area),
            )
        )

    sym_pairs = _find_symmetric_pairs(groups, cw, ch)

    # Silhouette info from G0
    sil_info = None
    if groups and groups[0].polygon:
        g0 = groups[0]
        g0_sd = _shape_descriptor(g0.polygon)
        if g0_sd:
            sil_info = SilhouetteInfo(
                shape_class=g0_sd["shape"],
                bbox=g0.polygon.bounds,
                aspect_ratio=g0_sd["aspect"],
                circularity=g0_sd["compactness"],
                convexity=g0_sd["convexity"],
            )

    composite_lines = _rasterize_composite(groups, cw, ch, grid_w=48)

    enrichment_text = build_enrichment_text(groups, silhouettes, cw, ch)

    return EnrichmentOutput(
        source="uploaded",
        canvas=(cw, ch),
        element_count=len(groups),
        subpath_count=n_raw,
        is_stroke_based=False,
        elements=elements,
        symmetry=SymmetryInfo(
            axis_type=_detect_axis_type(sym_pairs, groups, cw),
            score=_symmetry_score(sym_pairs, groups),
            pairs=[(_label(a), _label(b)) for a, b in sym_pairs],
        ),
        silhouette=sil_info,
        ascii_grid_positive="\n".join(composite_lines),
        enrichment_text=enrichment_text,
    )

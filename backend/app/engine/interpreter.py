"""Spatial Interpreter — synthesizes high-level clues from all 61 transforms.

Takes a completed PipelineContext and produces human-readable interpretation
that helps the LLM understand WHAT the SVG depicts, not just its geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.engine.context import PipelineContext


@dataclass
class SpatialInterpretation:
    """High-level spatial interpretation of an SVG."""

    summary: str = ""
    silhouette: str = ""
    orientation: str = ""
    composition_type: str = ""
    mass_distribution: str = ""
    focal_elements: list[str] = field(default_factory=list)
    protrusions: list[str] = field(default_factory=list)
    reading_hints: list[str] = field(default_factory=list)
    cluster_scene: list[str] = field(default_factory=list)
    # Global silhouette descriptors
    skeleton_desc: str = ""
    radial_profile: str = ""
    radial_features: str = ""
    contour_path: str = ""

    def to_text(self) -> str:
        lines = ["SPATIAL INTERPRETATION:"]
        if self.summary:
            lines.append(f"  LAYOUT SUMMARY: {self.summary}")
        if self.composition_type:
            lines.append(f"  Composition: {self.composition_type}")
        if self.orientation:
            lines.append(f"  Orientation: {self.orientation}")
        if self.silhouette:
            lines.append(f"  Silhouette: {self.silhouette}")
        if self.skeleton_desc:
            lines.append(f"  Skeleton: {self.skeleton_desc}")
        if self.radial_profile:
            lines.append(f"  {self.radial_profile}")
        if self.radial_features:
            lines.append(f"  {self.radial_features}")
        if self.contour_path:
            lines.append(f"  {self.contour_path}")
        if self.mass_distribution:
            lines.append(f"  Mass distribution: {self.mass_distribution}")
        if self.protrusions:
            lines.append("  Protrusions:")
            for a in self.protrusions:
                lines.append(f"    - {a}")
        if self.focal_elements:
            lines.append("  Focal elements:")
            for f in self.focal_elements:
                lines.append(f"    - {f}")
        if self.cluster_scene:
            lines.append("  Scene decomposition:")
            for c in self.cluster_scene:
                lines.append(f"    - {c}")
        if self.reading_hints:
            lines.append("  Reading hints:")
            for h in self.reading_hints:
                lines.append(f"    - {h}")
        return "\n".join(lines)


def interpret(ctx: PipelineContext) -> SpatialInterpretation:
    """Synthesize high-level spatial interpretation from pipeline context."""
    grid = _parse_grid(ctx.ascii_grid_positive)

    orientation = _interpret_orientation(ctx, grid)
    focal_elements = _identify_focal_elements(ctx)
    reading_hints = _generate_reading_hints(ctx, grid)

    # Global silhouette descriptors from composite grid
    skeleton_desc = ""
    radial_profile = ""
    radial_features = ""
    contour_path = ""

    if ctx.composite_grid is not None:
        from app.utils.rasterizer import (
            radial_distance_profile,
            simplified_contour_path,
            skeleton_description,
        )

        skeleton_desc = skeleton_description(ctx.composite_grid)
        radial_profile, radial_features = radial_distance_profile(
            ctx.composite_grid
        )
        contour_path = simplified_contour_path(
            ctx.composite_silhouette,
            ctx.composite_grid,
            ctx.canvas_width,
            ctx.canvas_height,
        )

    return SpatialInterpretation(
        summary=_generate_summary(ctx, orientation, focal_elements, reading_hints),
        silhouette=_interpret_silhouette(ctx, grid),
        orientation=orientation,
        composition_type=_interpret_composition(ctx),
        mass_distribution=_analyze_mass_distribution(grid),
        focal_elements=focal_elements,
        protrusions=_detect_protrusions_from_elements(ctx, grid),
        reading_hints=reading_hints,
        cluster_scene=_decompose_clusters(ctx),
        skeleton_desc=skeleton_desc,
        radial_profile=radial_profile,
        radial_features=radial_features,
        contour_path=contour_path,
    )


def _generate_summary(
    ctx: PipelineContext,
    orientation: str,
    focal_elements: list[str],
    reading_hints: list[str],
) -> str:
    """Generate a concise layout summary synthesizing orientation + structure.

    Describes composition shape, dominant axis, structural layering, and mass
    distribution in neutral geometric terms.
    """
    import re

    parts = []

    # Orientation summary
    if "strong bilateral symmetry" in orientation.lower():
        parts.append("symmetric composition (strong bilateral)")
    elif "asymmetric" in orientation.lower():
        # Extract mass direction
        if "mass concentrated right" in orientation.lower():
            parts.append("asymmetric composition (mass toward right)")
        elif "mass concentrated left" in orientation.lower():
            parts.append("asymmetric composition (mass toward left)")
        else:
            parts.append("asymmetric composition")

    # Mass distribution from grid
    grid = _parse_grid(ctx.ascii_grid_positive)
    if grid:
        row_profile = _row_profile(grid)
        if row_profile and len(row_profile) >= 6:
            n_rows = len(row_profile)
            third = n_rows // 3
            top_fill = sum(row_profile[:third])
            mid_fill = sum(row_profile[third:2*third])
            bot_fill = sum(row_profile[2*third:])
            total = top_fill + mid_fill + bot_fill
            if total > 0:
                bot_pct = bot_fill / total
                mid_pct = mid_fill / total
                top_pct = top_fill / total
                if bot_pct > 0.42:
                    parts.append("mass concentrated at bottom")
                elif top_pct > 0.42:
                    parts.append("mass concentrated at top")
                elif mid_pct > 0.42:
                    parts.append("mass concentrated at middle")

    # Count structural layers from reading hints
    layer_count = 0
    for hint in reading_hints:
        match = re.search(r"Major structural elements.*?:\s*(.*)", hint)
        if match:
            layer_count = match.group(1).count(";") + 1

    if layer_count >= 2:
        parts.append(f"{layer_count} structural layers")

    # Background element info
    bg_area_pct = ""
    bg_extends_up = False
    bg_wide = False
    for hint in reading_hints:
        if "LARGER than primary boundary" in hint:
            match = re.search(r"area=(\d+)% of primary", hint)
            if match:
                bg_area_pct = match.group(1) + "%"
        if "Background element layout" in hint:
            if "extends up" in hint.lower() or "upper half" in hint.lower():
                bg_extends_up = True
            if "spans" in hint.lower() and "%" in hint:
                bg_wide = True

    bg_parts = []
    if bg_extends_up:
        bg_parts.append("extends into upper half")
    if bg_wide:
        bg_parts.append("spans wide")
    if bg_area_pct:
        bg_parts.append(f"area={bg_area_pct} of primary (LARGER than primary boundary)")

    if bg_parts:
        parts.append(f"large background element: {', '.join(bg_parts)}")

    # Focal element offset from center
    for fe in focal_elements:
        if "offset from center" in fe.lower():
            parts.append(fe.strip())
            break

    if not parts:
        return ""

    return "; ".join(parts)


# ── Grid parsing ──


def _parse_grid(grid_text: str) -> list[list[bool]]:
    """Parse ASCII grid into 2D boolean array. X=True, .=False."""
    if not grid_text:
        return []
    rows = []
    for line in grid_text.strip().split("\n"):
        row = []
        for ch in line:
            if ch == "X":
                row.append(True)
            elif ch == ".":
                row.append(False)
            # skip spaces
        rows.append(row)
    return rows


def _grid_stats(grid: list[list[bool]]) -> dict:
    """Compute basic grid statistics."""
    if not grid:
        return {"rows": 0, "cols": 0, "fill_pct": 0.0}

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    total = 0
    filled = 0

    for row in grid:
        for cell in row:
            total += 1
            if cell:
                filled += 1

    return {
        "rows": rows,
        "cols": cols,
        "total": total,
        "filled": filled,
        "fill_pct": (filled / total * 100) if total > 0 else 0.0,
    }


def _quadrant_fill(grid: list[list[bool]]) -> dict[str, float]:
    """Compute fill percentage per quadrant (UL, UR, LL, LR)."""
    if not grid:
        return {"UL": 0, "UR": 0, "LL": 0, "LR": 0}

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    mid_r, mid_c = rows // 2, cols // 2

    quads = {"UL": [0, 0], "UR": [0, 0], "LL": [0, 0], "LR": [0, 0]}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if r < mid_r:
                q = "UL" if c < mid_c else "UR"
            else:
                q = "LL" if c < mid_c else "LR"
            quads[q][1] += 1
            if cell:
                quads[q][0] += 1

    return {
        k: (v[0] / v[1] * 100 if v[1] > 0 else 0.0)
        for k, v in quads.items()
    }


def _column_profile(grid: list[list[bool]]) -> list[float]:
    """Fill percentage per column (left-to-right mass profile)."""
    if not grid:
        return []
    cols = max(len(r) for r in grid)
    profile = []
    for c in range(cols):
        filled = sum(1 for row in grid if c < len(row) and row[c])
        profile.append(filled / len(grid) * 100)
    return profile


def _row_profile(grid: list[list[bool]]) -> list[float]:
    """Fill percentage per row (top-to-bottom mass profile)."""
    if not grid:
        return []
    cols = max(len(r) for r in grid) if grid else 1
    profile = []
    for row in grid:
        filled = sum(1 for cell in row if cell)
        profile.append(filled / cols * 100)
    return profile


def _find_protrusions(grid: list[list[bool]]) -> list[dict]:
    """Find regions that protrude from the main mass."""
    if not grid:
        return []

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    col_profile = _column_profile(grid)
    row_profile = _row_profile(grid)

    # Median fill gives us the "core" density
    col_median = float(np.median(col_profile)) if col_profile else 50.0
    row_median = float(np.median(row_profile)) if row_profile else 50.0

    protrusions = []

    # Check top edge — find columns with fill in top rows but not in main body
    top_extent = 0
    for r in range(rows):
        if row_profile[r] > 20:
            top_extent = r
            break

    if top_extent > 2:
        # Top protrusion: find which columns have fill in the top region
        top_cols_filled = []
        for c in range(cols):
            top_fill = sum(1 for r in range(top_extent) if r < rows and c < len(grid[r]) and grid[r][c])
            if top_fill > 0:
                top_cols_filled.append(c)
        if top_cols_filled:
            region = "left" if np.mean(top_cols_filled) < cols / 2 else "right"
            if abs(np.mean(top_cols_filled) - cols / 2) < cols * 0.15:
                region = "center"
            protrusions.append({
                "direction": "upward",
                "region": region,
                "extent": top_extent,
                "width_cols": len(top_cols_filled),
            })

    # Check bottom edge
    bottom_extent = 0
    for r in range(rows - 1, -1, -1):
        if row_profile[r] > 20:
            bottom_extent = rows - 1 - r
            break

    if bottom_extent > 2:
        bottom_cols_filled = []
        for c in range(cols):
            bottom_fill = sum(
                1 for r in range(rows - bottom_extent, rows)
                if c < len(grid[r]) and grid[r][c]
            )
            if bottom_fill > 0:
                bottom_cols_filled.append(c)
        if bottom_cols_filled:
            region = "left" if np.mean(bottom_cols_filled) < cols / 2 else "right"
            if abs(np.mean(bottom_cols_filled) - cols / 2) < cols * 0.15:
                region = "center"
            protrusions.append({
                "direction": "downward",
                "region": region,
                "extent": bottom_extent,
            })

    # Check left edge
    left_sparse = sum(1 for c in range(min(4, cols)) if col_profile[c] < 30)
    right_sparse = sum(1 for c in range(max(0, cols - 4), cols) if col_profile[c] < 30)

    if left_sparse > 2:
        protrusions.append({"direction": "left_sparse", "note": "left edge has low fill"})
    if right_sparse > 2:
        protrusions.append({"direction": "right_sparse", "note": "right edge has low fill"})

    return protrusions


def _compute_principal_axis(grid: list[list[bool]]) -> dict:
    """Compute principal axis of the filled mass using 2nd-moment (covariance).

    Returns angle in degrees (0°=horizontal, 90°=vertical) and eccentricity.
    Eccentricity near 1.0 = strongly oriented; near 0.0 = roughly circular mass.
    """
    if not grid:
        return {"angle": 0.0, "eccentricity": 0.0, "orientation": "unknown"}

    # Collect filled cell coordinates
    coords = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell:
                coords.append((c, r))  # (x, y)

    if len(coords) < 3:
        return {"angle": 0.0, "eccentricity": 0.0, "orientation": "unknown"}

    arr = np.array(coords, dtype=float)
    # Center of mass
    cx = np.mean(arr[:, 0])
    cy = np.mean(arr[:, 1])

    # Central moments (covariance matrix)
    dx = arr[:, 0] - cx
    dy = arr[:, 1] - cy
    mu20 = np.mean(dx * dx)
    mu02 = np.mean(dy * dy)
    mu11 = np.mean(dx * dy)

    # Eigenvalues of covariance matrix → principal axes
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11 * mu11
    discriminant = max(0, trace * trace / 4 - det)
    sqrt_disc = discriminant ** 0.5
    lambda1 = trace / 2 + sqrt_disc  # larger eigenvalue
    lambda2 = trace / 2 - sqrt_disc  # smaller eigenvalue

    # Angle of principal axis (dominant direction of spread)
    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = float(np.degrees(angle_rad))

    # Eccentricity: how elongated the mass distribution is
    if lambda1 > 0:
        eccentricity = float(1.0 - lambda2 / lambda1)
    else:
        eccentricity = 0.0

    # Classify orientation
    # angle_deg is in [-90, 90]. Near 0 = horizontal spread, near ±90 = vertical spread.
    abs_angle = abs(angle_deg)
    if abs_angle > 60:
        orientation = "vertical"
    elif abs_angle < 30:
        orientation = "horizontal"
    else:
        orientation = "diagonal"

    return {
        "angle": angle_deg,
        "eccentricity": eccentricity,
        "orientation": orientation,
    }


def _compute_boundary_lobes(grid: list[list[bool]]) -> dict:
    """Compute the centroid distance function and count boundary lobes.

    Samples the distance from the centroid to the boundary at regular angles.
    Local maxima in this function correspond to "lobes" — distinct protruding
    mass regions (head, tail, limbs, etc.). Multi-lobed = complex silhouette.
    """
    if not grid:
        return {"lobe_count": 0, "max_min_ratio": 1.0}

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    if rows < 6 or cols < 6:
        return {"lobe_count": 0, "max_min_ratio": 1.0}

    # Find centroid of filled region
    filled_coords = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell:
                filled_coords.append((c, r))

    if len(filled_coords) < 10:
        return {"lobe_count": 0, "max_min_ratio": 1.0}

    arr = np.array(filled_coords, dtype=float)
    cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))

    # Sample distance from centroid to boundary at N angles
    n_angles = 72  # every 5 degrees
    distances = []
    for i in range(n_angles):
        angle = 2 * np.pi * i / n_angles
        dx, dy = np.cos(angle), np.sin(angle)
        # Walk outward from centroid until hitting empty space or boundary
        max_dist = 0
        for step in range(1, max(rows, cols)):
            px = int(cx + dx * step)
            py = int(cy + dy * step)
            if px < 0 or px >= cols or py < 0 or py >= rows:
                max_dist = step
                break
            if py < len(grid) and px < len(grid[py]) and grid[py][px]:
                max_dist = step
            else:
                break
        distances.append(max_dist)

    if not distances or max(distances) == 0:
        return {"lobe_count": 0, "max_min_ratio": 1.0}

    # Smooth to reduce noise
    kernel = np.ones(5) / 5
    # Circular smoothing: pad the signal
    padded = distances + distances + distances
    smoothed_full = np.convolve(padded, kernel, mode='same')
    smoothed = smoothed_full[n_angles:2 * n_angles]

    # Count local maxima (lobes) — peaks that are at least 20% above surrounding valleys
    min_dist = min(smoothed)
    max_dist = max(smoothed)
    threshold = min_dist + (max_dist - min_dist) * 0.2

    lobes = 0
    in_peak = False
    for i in range(n_angles):
        if smoothed[i] > threshold and not in_peak:
            in_peak = True
            lobes += 1
        elif smoothed[i] <= threshold:
            in_peak = False

    ratio = float(max_dist / min_dist) if min_dist > 0 else float('inf')

    return {"lobe_count": lobes, "max_min_ratio": ratio}


def _compute_convex_hull_deficiency(grid: list[list[bool]]) -> float:
    """Ratio of filled area to convex hull area. Low = many protrusions/concavities."""
    if not grid:
        return 1.0

    coords = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell:
                coords.append((c, r))

    if len(coords) < 3:
        return 1.0

    from scipy.spatial import ConvexHull

    try:
        arr = np.array(coords, dtype=float)
        hull = ConvexHull(arr)
        hull_area = hull.volume  # In 2D, ConvexHull.volume = area
        filled_area = len(coords)
        return float(filled_area / hull_area) if hull_area > 0 else 1.0
    except Exception:
        return 1.0


# ── Interpretation functions ──


def _interpret_silhouette(ctx: PipelineContext, grid: list[list[bool]]) -> str:
    """Describe the overall silhouette shape from the positive space grid."""
    if not grid:
        return "no grid data available"

    stats = _grid_stats(grid)
    quads = _quadrant_fill(grid)
    col_profile = _column_profile(grid)
    row_profile = _row_profile(grid)

    # Overall shape
    fill = stats["fill_pct"]
    parts = []

    if fill > 85:
        parts.append("dense, nearly rectangular fill")
    elif fill > 65:
        parts.append("mostly filled with shaped contour")
    elif fill > 40:
        parts.append("moderate fill with clear negative space")
    else:
        parts.append("sparse, open composition")

    # Widest/narrowest points — skip for very dense fills where widest is meaningless
    if col_profile and fill < 80:
        widest_row = int(np.argmax(row_profile))
        narrowest_filled = [i for i, v in enumerate(row_profile) if v > 10]
        if narrowest_filled:
            row_count = len(row_profile)
            widest_pos = "top" if widest_row < row_count * 0.33 else ("middle" if widest_row < row_count * 0.66 else "bottom")
            parts.append(f"widest at {widest_pos}")
    elif col_profile and fill >= 80:
        # Dense fill — describe where the silhouette NARROWS or has gaps instead
        row_count = len(row_profile)
        # Find which rows have the least fill (that's where the silhouette shape is)
        bottom_third_fill = np.mean(row_profile[2 * row_count // 3:]) if row_count > 3 else 100
        top_third_fill = np.mean(row_profile[:row_count // 3]) if row_count > 3 else 100
        if bottom_third_fill < top_third_fill - 15:
            parts.append("narrows toward bottom")
        elif top_third_fill < bottom_third_fill - 15:
            parts.append("narrows toward top")

    # Vertical extent vs horizontal
    filled_rows = [i for i, v in enumerate(row_profile) if v > 10]
    filled_cols = [i for i, v in enumerate(col_profile) if v > 10]
    if filled_rows and filled_cols:
        v_span = filled_rows[-1] - filled_rows[0] + 1
        h_span = filled_cols[-1] - filled_cols[0] + 1
        if v_span > h_span * 1.3:
            parts.append("taller than wide")
        elif h_span > v_span * 1.3:
            parts.append("wider than tall")
        else:
            parts.append("roughly square proportions")

    # Asymmetry
    left_fill = sum(col_profile[:len(col_profile) // 2])
    right_fill = sum(col_profile[len(col_profile) // 2:])
    total_fill = left_fill + right_fill
    if total_fill > 0:
        balance = left_fill / total_fill
        if balance > 0.6:
            parts.append("heavier on left side")
        elif balance < 0.4:
            parts.append("heavier on right side")

    # Protrusions
    protrusions = _find_protrusions(grid)
    for p in protrusions:
        if p["direction"] == "upward":
            parts.append(f"upward protrusion from {p['region']}")
        elif p["direction"] == "downward":
            parts.append(f"extends down on {p['region']}")

    # Width profile: find narrowest point ("waist") and distinct mass regions
    if row_profile and len(row_profile) >= 10:
        # Smooth the profile to avoid noise
        kernel_size = max(3, len(row_profile) // 10)
        smoothed = np.convolve(row_profile, np.ones(kernel_size) / kernel_size, mode='same')
        # Find local minima (valleys) in the smoothed profile where fill > 5%
        # Only consider rows that are within the filled region
        filled_range = [i for i, v in enumerate(smoothed) if v > 5]
        if len(filled_range) >= 6:
            start_r, end_r = filled_range[0], filled_range[-1]
            inner = smoothed[start_r:end_r + 1]
            if len(inner) >= 6:
                # Find the deepest valley (narrowest point between two wider regions)
                min_val = float('inf')
                min_pos = 0
                # Only look between 20% and 80% of the filled region
                search_start = len(inner) // 5
                search_end = 4 * len(inner) // 5
                for i in range(search_start, search_end):
                    if inner[i] < min_val:
                        min_val = inner[i]
                        min_pos = i
                # Check if there's a real "waist" — the narrowest point must be notably narrower
                max_above = max(inner[:min_pos]) if min_pos > 0 else 0
                max_below = max(inner[min_pos:]) if min_pos < len(inner) else 0
                wider = min(max_above, max_below)
                if wider > 0 and min_val < wider * 0.7:
                    waist_row = start_r + min_pos
                    waist_pct = waist_row / len(row_profile) * 100
                    parts.append(
                        f"narrowest at row {waist_pct:.0f}% (width drops to {min_val:.0f}% vs {wider:.0f}% above/below) — distinct upper and lower mass regions"
                    )

    # Empty corners/edges — tells the LLM where the figure DOESN'T extend
    quads = _quadrant_fill(grid)
    quad_names = {"UL": "upper-left", "UR": "upper-right", "LL": "lower-left", "LR": "lower-right"}
    empty_quads = [quad_names[k] for k, v in quads.items() if v < 50]
    sparse_quads = [quad_names[k] for k, v in quads.items() if 50 <= v < 70]
    if empty_quads:
        parts.append(f"empty: {', '.join(empty_quads)}")
    if sparse_quads:
        parts.append(f"sparse: {', '.join(sparse_quads)}")

    # Bottom/top edge readings — which side of the bottom has fill
    # This helps distinguish sitting figures, grounded objects, etc.
    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    if rows > 6 and cols > 6:
        bottom_rows = grid[max(0, rows - 4):]
        left_bottom = sum(
            1 for r in bottom_rows for c in range(min(cols // 2, len(r))) if c < len(r) and r[c]
        )
        right_bottom = sum(
            1 for r in bottom_rows for c in range(cols // 2, min(cols, len(r))) if c < len(r) and r[c]
        )
        total_bottom = left_bottom + right_bottom
        if total_bottom > 0:
            left_pct = left_bottom / total_bottom
            if left_pct < 0.25:
                parts.append("bottom edge: right side only (left side open)")
            elif left_pct > 0.75:
                parts.append("bottom edge: left side only (right side open)")
            elif left_pct < 0.4:
                parts.append("bottom edge: mostly right")
            elif left_pct > 0.6:
                parts.append("bottom edge: mostly left")

    # Principal axis of mass — tells you the dominant orientation of the shape
    paxis = _compute_principal_axis(grid)
    if paxis["orientation"] != "unknown":
        parts.append(
            f"principal axis: {paxis['orientation']} ({paxis['angle']:.0f}°, "
            f"eccentricity={paxis['eccentricity']:.2f})"
        )

    # Convex hull deficiency — how much the shape deviates from its convex hull
    hull_def = _compute_convex_hull_deficiency(grid)
    if hull_def < 0.7:
        parts.append(f"highly concave silhouette (hull fill={hull_def:.0%})")
    elif hull_def < 0.85:
        parts.append(f"moderately concave (hull fill={hull_def:.0%})")

    # Boundary lobe analysis — how many distinct mass regions protrude from center
    lobes = _compute_boundary_lobes(grid)
    if lobes["lobe_count"] >= 2:
        parts.append(
            f"boundary lobes: {lobes['lobe_count']} distinct protruding regions "
            f"(max/min distance ratio={lobes['max_min_ratio']:.1f})"
        )

    return "; ".join(parts)


def _interpret_orientation(ctx: PipelineContext, grid: list[list[bool]]) -> str:
    """Determine orientation from symmetry and mass distribution."""
    sym_score = ctx.symmetry_score
    sym_axis = ctx.symmetry_axis or "none"

    # Mass balance from grid
    col_profile = _column_profile(grid)
    if col_profile:
        mid = len(col_profile) // 2
        left_mass = sum(col_profile[:mid])
        right_mass = sum(col_profile[mid:])
        total = left_mass + right_mass
        balance = left_mass / total if total > 0 else 0.5
    else:
        balance = 0.5

    parts = []

    # Symmetry interpretation
    if sym_score >= 0.75:
        parts.append("strong bilateral symmetry")
        parts.append(f"strong {sym_axis} symmetry ({sym_score:.2f})")
    elif sym_score >= 0.5:
        parts.append("semi-symmetric")
        if abs(balance - 0.5) > 0.1:
            side = "left" if balance > 0.55 else "right"
            parts.append(f"mass concentrated {side}")
        else:
            parts.append("roughly centered but with asymmetric features")
        parts.append(f"moderate {sym_axis} symmetry ({sym_score:.2f})")
    elif sym_score >= 0.25:
        if abs(balance - 0.5) > 0.15:
            side = "left" if balance > 0.55 else "right"
            parts.append(f"asymmetric, mass concentrated {side}")
        else:
            parts.append("asymmetric composition")
        parts.append(f"weak symmetry ({sym_score:.2f})")
    else:
        parts.append("highly asymmetric / abstract")
        parts.append(f"minimal symmetry ({sym_score:.2f})")

    return "; ".join(parts)


def _interpret_composition(ctx: PipelineContext) -> str:
    """Classify the composition type based on element count and structure."""
    n = len(ctx.subpaths)

    # Nesting depth from containment
    max_depth = 0
    if ctx.containment_matrix is not None:
        for i in range(n):
            depth = sum(1 for j in range(n) if j != i and ctx.containment_matrix[j][i])
            max_depth = max(max_depth, depth)

    # Shape variety
    shape_classes = set()
    for sp in ctx.subpaths:
        shape_classes.add(sp.features.get("shape_class", "organic"))

    # Classify
    if n <= 5:
        comp = "simple icon"
    elif n <= 15:
        comp = "structured icon or logo"
    elif n <= 40:
        comp = "detailed icon or illustration"
    else:
        comp = "complex illustration"

    if max_depth >= 3:
        comp += f" with deep nesting ({max_depth} levels)"
    elif max_depth >= 2:
        comp += " with nested elements"

    if len(shape_classes) >= 4:
        comp += f", mixed shapes ({', '.join(sorted(shape_classes))})"
    elif len(shape_classes) == 1:
        comp += f", uniform {list(shape_classes)[0]} shapes"

    return comp


def _analyze_mass_distribution(grid: list[list[bool]]) -> str:
    """Describe where visual weight is concentrated."""
    if not grid:
        return "unknown"

    quads = _quadrant_fill(grid)
    col_profile = _column_profile(grid)
    row_profile = _row_profile(grid)

    parts = []

    # Find dominant quadrant(s)
    max_fill = max(quads.values())
    dominant = [k for k, v in quads.items() if v > max_fill * 0.85]

    quad_names = {"UL": "upper-left", "UR": "upper-right", "LL": "lower-left", "LR": "lower-right"}
    if len(dominant) == 4:
        parts.append("evenly distributed")
    elif len(dominant) <= 2:
        parts.append(f"concentrated in {', '.join(quad_names[d] for d in dominant)}")
    else:
        parts.append("broadly distributed")

    # Vertical center of mass
    if row_profile:
        total_mass = sum(row_profile)
        if total_mass > 0:
            com_row = sum(i * v for i, v in enumerate(row_profile)) / total_mass
            row_pos = com_row / len(row_profile)
            if row_pos < 0.4:
                parts.append("vertically top-heavy")
            elif row_pos > 0.6:
                parts.append("vertically bottom-heavy")
            else:
                parts.append("vertically centered")

    # Horizontal center of mass
    if col_profile:
        total_mass = sum(col_profile)
        if total_mass > 0:
            com_col = sum(i * v for i, v in enumerate(col_profile)) / total_mass
            col_pos = com_col / len(col_profile)
            if col_pos < 0.4:
                parts.append("horizontally left-leaning")
            elif col_pos > 0.6:
                parts.append("horizontally right-leaning")

    # Fill percentage per quadrant
    quad_strs = [f"{quad_names[k]}={v:.0f}%" for k, v in quads.items()]
    parts.append(f"quadrants: {', '.join(quad_strs)}")

    # Vertical thirds — where is the mass concentrated?
    if row_profile and len(row_profile) >= 6:
        n_rows = len(row_profile)
        third = n_rows // 3
        top_fill = sum(row_profile[:third])
        mid_fill = sum(row_profile[third:2*third])
        bot_fill = sum(row_profile[2*third:])
        total = top_fill + mid_fill + bot_fill
        if total > 0:
            top_pct = top_fill / total * 100
            mid_pct = mid_fill / total * 100
            bot_pct = bot_fill / total * 100
            parts.append(f"vertical thirds: top={top_pct:.0f}%, mid={mid_pct:.0f}%, bottom={bot_pct:.0f}%")

    return "; ".join(parts)


def _identify_focal_elements(ctx: PipelineContext) -> list[str]:
    """Find elements that likely represent key visual features."""
    focal = []

    canvas_w, canvas_h = ctx.canvas_width, ctx.canvas_height

    # Concentric groups — multi-ring constructions (most structurally significant)
    concentric = ctx.subpaths[0].features.get("concentric_groups", []) if ctx.subpaths else []
    concentric_entries: list[tuple[float, tuple[float, float], str]] = []  # (circ, center, text)
    for group in concentric:
        members = group.get("members", [])
        center = group.get("center", (0, 0))
        if len(members) >= 2:
            member_sps = [sp for sp in ctx.subpaths if sp.id in members]
            circs = [sp.features.get("circularity", 0) for sp in member_sps]
            max_circ = max(circs) if circs else 0
            if max_circ > 0.5:
                tiers = [sp.features.get("size_tier", "MEDIUM") for sp in member_sps]
                # Describe position on canvas
                pos_x = "left" if center[0] < canvas_w * 0.33 else ("center" if center[0] < canvas_w * 0.66 else "right")
                pos_y = "top" if center[1] < canvas_h * 0.33 else ("mid" if center[1] < canvas_h * 0.66 else "bottom")

                # Containment context — which structural element contains this group?
                parent_info = ""
                if ctx.containment_matrix is not None:
                    first_id = members[0]
                    first_idx = next((i for i, s in enumerate(ctx.subpaths) if s.id == first_id), -1)
                    if first_idx >= 0:
                        member_set = set(members)
                        parents = [
                            i for i in range(len(ctx.subpaths))
                            if i != first_idx
                            and ctx.subpaths[i].id not in member_set
                            and ctx.containment_matrix[i][first_idx]
                        ]
                        if parents:
                            parent_idx = min(parents, key=lambda i: ctx.subpaths[i].area)
                            parent_info = f", inside {ctx.subpaths[parent_idx].id}"

                # Edge annotation — groups at canvas edge are likely decorative
                at_edge = (center[0] < 5 or center[0] > canvas_w - 5 or
                           center[1] < 5 or center[1] > canvas_h - 5)
                if at_edge:
                    edge_info = ", at canvas edge"
                elif not parent_info:
                    edge_info = ", root-level (independent, drawn on top)"
                else:
                    edge_info = ""

                entry = (
                    f"Concentric group [{', '.join(members)}] at ({center[0]:.0f},{center[1]:.0f}) [{pos_y}-{pos_x}], "
                    f"{len(members)} layers, max circularity={max_circ:.2f}, tiers={'/'.join(tiers)}{parent_info}{edge_info}"
                )
                concentric_entries.append((max_circ, center, entry))
    # Show top 4 by circularity
    concentric_entries.sort(key=lambda x: x[0], reverse=True)
    for _, _, entry in concentric_entries[:4]:
        focal.append(entry)

    # Add spatial relationships from the most circular group to others
    if len(concentric_entries) >= 2:
        # Only show relationships from the top group (highest circularity) to others
        # Skip groups near canvas edges (likely decorative, not structural)
        top = concentric_entries[0]
        relations = []
        for entry in concentric_entries[1:4]:
            c1 = top[1]
            c2 = entry[1]
            # Skip if either group is at canvas edge (within 5% of border)
            edge_margin = min(canvas_w, canvas_h) * 0.05
            if (c2[0] < edge_margin or c2[0] > canvas_w - edge_margin) and \
               (c2[1] < edge_margin or c2[1] > canvas_h - edge_margin):
                continue
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            if abs(dy) < canvas_h * 0.1:
                direction = "right of" if dx > 0 else "left of"
            elif abs(dx) < canvas_w * 0.1:
                direction = "below" if dy > 0 else "above"
            else:
                v = "below" if dy > 0 else "above"
                h = "right" if dx > 0 else "left"
                direction = f"{v}-{h} of"
            dist = (dx**2 + dy**2) ** 0.5
            relations.append(
                f"  -> group at ({c2[0]:.0f},{c2[1]:.0f}) is {direction} group at ({c1[0]:.0f},{c1[1]:.0f}), {dist:.0f}px apart"
            )
        focal.extend(relations[:3])

    # Individual high-circularity elements (not already covered by concentric groups)
    concentric_ids = set()
    for group in concentric:
        concentric_ids.update(group.get("members", []))

    for sp in ctx.subpaths:
        if sp.id in concentric_ids:
            continue
        circ = sp.features.get("circularity", 0.0)
        tier = sp.features.get("size_tier", "MEDIUM")
        shape = sp.features.get("shape_class", "organic")
        cx, cy = sp.centroid

        if circ > 0.75 and tier in ("SMALL", "MEDIUM"):
            contained_by = sp.features.get("contained_by", [])
            if contained_by:
                focal.append(
                    f"{sp.id}: circular ({shape}, circ={circ:.2f}) inside {contained_by[0]}"
                )
            else:
                focal.append(
                    f"{sp.id}: circular ({shape}, circ={circ:.2f}) at ({cx:.0f},{cy:.0f}), standalone"
                )

    # Offset indicator: if the highest-circularity root-level concentric group
    # is far from canvas center, note the offset.
    if concentric_entries and canvas_w > 0:
        for circ_val, center, entry_text in concentric_entries:
            at_edge = (center[0] < 5 or center[0] > canvas_w - 5 or
                       center[1] < 5 or center[1] > canvas_h - 5)
            if at_edge:
                continue
            if "root-level" in entry_text or "inside" not in entry_text:
                x_pct = center[0] / canvas_w
                if x_pct > 0.75 or x_pct < 0.25:
                    side = "right" if x_pct > 0.5 else "left"
                    focal.append(
                        f"Highest-circularity concentric group at x={x_pct:.0%} — "
                        f"offset from center (far {side})"
                    )
                break

    # Limit to most interesting
    return focal[:12]


def _detect_protrusions_from_elements(ctx: PipelineContext, grid: list[list[bool]]) -> list[str]:
    """Detect elements that protrude from the core mass."""
    results = []

    if not ctx.subpaths or not grid:
        return results

    canvas_w, canvas_h = ctx.canvas_width, ctx.canvas_height

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    if rows == 0 or cols == 0:
        return results

    row_fills = _row_profile(grid)
    col_fills = _column_profile(grid)

    core_rows = [i for i, v in enumerate(row_fills) if v > 50]
    core_cols = [i for i, v in enumerate(col_fills) if v > 50]

    if not core_rows or not core_cols:
        return results

    core_top = core_rows[0] / rows * canvas_h
    core_bottom = core_rows[-1] / rows * canvas_h
    core_left = core_cols[0] / cols * canvas_w
    core_right = core_cols[-1] / cols * canvas_w
    margin_x = canvas_w * 0.08
    margin_y = canvas_h * 0.08

    for sp in ctx.subpaths:
        cx, cy = sp.centroid
        tier = sp.features.get("size_tier", "MEDIUM")
        ar = sp.features.get("aspect_ratio", 1.0)

        if cy < core_top - margin_y:
            desc = f"{sp.id}: [{tier}] protrudes above core mass at ({cx:.0f},{cy:.0f})"
            if ar > 1.5:
                desc += f", wide (aspect={ar:.1f})"
            elif ar < 0.6:
                desc += f", tall (aspect={ar:.1f})"
            results.append(desc)
        elif cy > core_bottom + margin_y:
            results.append(f"{sp.id}: [{tier}] protrudes below core mass at ({cx:.0f},{cy:.0f})")
        elif cx < core_left - margin_x:
            results.append(f"{sp.id}: [{tier}] protrudes left of core mass at ({cx:.0f},{cy:.0f})")
        elif cx > core_right + margin_x:
            results.append(f"{sp.id}: [{tier}] protrudes right of core mass at ({cx:.0f},{cy:.0f})")

    return results[:10]


def _generate_reading_hints(ctx: PipelineContext, grid: list[list[bool]]) -> list[str]:
    """Generate plain-language hints for interpreting the SVG."""
    hints = []
    n = len(ctx.subpaths)

    # Symmetry hint
    sym_score = ctx.symmetry_score
    if sym_score < 0.5:
        hints.append(
            f"Low symmetry ({sym_score:.2f}) — asymmetric composition, side-view, "
            "or angled arrangement."
        )
    elif sym_score > 0.8:
        hints.append(
            f"High symmetry ({sym_score:.2f}) — balanced/symmetric design or "
            "front-facing orientation."
        )

    # Mass distribution hint
    quads = _quadrant_fill(grid)
    if quads:
        imbalance = max(quads.values()) - min(quads.values())
        if imbalance > 30:
            heavy = max(quads, key=quads.get)
            quad_names = {"UL": "upper-left", "UR": "upper-right", "LL": "lower-left", "LR": "lower-right"}
            hints.append(
                f"Visual weight is concentrated {quad_names[heavy]} "
                f"({quads[heavy]:.0f}% fill vs {min(quads.values()):.0f}% in lightest quadrant)."
            )

    # Containment depth
    if ctx.containment_matrix is not None:
        max_depth = 0
        for i in range(n):
            depth = sum(1 for j in range(n) if j != i and ctx.containment_matrix[j][i])
            max_depth = max(max_depth, depth)
        if max_depth >= 3:
            hints.append(f"Deep nesting ({max_depth} levels).")

    # Structural hierarchy — major top-level elements from stacking tree
    if ctx.subpaths and n > 15:
        tree_text = ctx.subpaths[0].features.get("stacking_tree_text", "")
        if tree_text:
            root_ids = []
            for line in tree_text.strip().split("\n"):
                if line and not line[0].isspace():
                    eid = line.split(" ")[0].strip()
                    if eid.startswith("E"):
                        root_ids.append(eid)

            large_roots = []
            for rid in root_ids:
                sp = next((s for s in ctx.subpaths if s.id == rid), None)
                if sp and sp.features.get("size_tier") == "LARGE":
                    x1, y1, x2, y2 = sp.bbox
                    cov_w = (x2 - x1) / ctx.canvas_width * 100 if ctx.canvas_width > 0 else 0
                    cov_h = (y2 - y1) / ctx.canvas_height * 100 if ctx.canvas_height > 0 else 0

                    if cov_h > 85 and cov_w > 85:
                        extent = "spans full canvas"
                    elif cov_h > 60 and cov_w > 60:
                        extent = f"covers most of canvas ({cov_w:.0f}%w x {cov_h:.0f}%h)"
                    else:
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        h_pos = "left" if cx < ctx.canvas_width * 0.33 else ("center" if cx < ctx.canvas_width * 0.66 else "right")
                        v_pos = "top" if cy < ctx.canvas_height * 0.33 else ("mid" if cy < ctx.canvas_height * 0.66 else "bottom")
                        extent = f"{v_pos}-{h_pos}, {cov_w:.0f}%w x {cov_h:.0f}%h"

                    child_count = 0
                    z_index = -1
                    if ctx.containment_matrix is not None:
                        idx = next((i for i, s in enumerate(ctx.subpaths) if s.id == rid), -1)
                        if idx >= 0:
                            z_index = idx
                            child_count = sum(
                                1 for j in range(n)
                                if j != idx and ctx.containment_matrix[idx][j]
                            )

                    # Shape descriptor with precise measurements
                    shape_desc = ""
                    if sp:
                        conv = sp.features.get("convexity", 0)
                        asp = sp.features.get("aspect_ratio", 1.0)
                        sc = sp.features.get("shape_class", "organic")
                        if sc == "circular":
                            shape_desc = f"circular (conv={conv:.2f}, aspect={asp:.1f})"
                        elif conv > 0.85:
                            shape_desc = f"compact rounded (conv={conv:.2f}, aspect={asp:.1f})"
                        elif conv > 0.6:
                            if asp < 2.0:
                                shape_desc = f"curved organic (conv={conv:.2f}, aspect={asp:.1f})"
                            else:
                                shape_desc = f"elongated (conv={conv:.2f}, aspect={asp:.1f})"
                        elif conv > 0.3:
                            if asp > 1.5:
                                shape_desc = f"irregular spread (conv={conv:.2f}, aspect={asp:.1f})"
                            else:
                                shape_desc = f"irregular (conv={conv:.2f}, aspect={asp:.1f})"
                        else:
                            shape_desc = f"thin/branching (conv={conv:.2f}, aspect={asp:.1f})"

                    large_roots.append((rid, extent, child_count, z_index, shape_desc))

            if len(large_roots) >= 2:
                # Sort by z-index so draw order is clear
                large_roots.sort(key=lambda x: x[3])
                # Find the primary boundary (full canvas or highest z)
                primary = next(
                    (r for r in large_roots if r[1] == "spans full canvas"),
                    large_roots[-1]  # fallback to highest z
                )
                primary_z = primary[3]
                primary_rid = primary[0]
                primary_sp = next((s for s in ctx.subpaths if s.id == primary_rid), None)
                primary_area = primary_sp.area if primary_sp else 1.0
                descs = []
                for rid, ext, ch, z, shape in large_roots[:6]:
                    z_label = f"z={z}" if z >= 0 else ""
                    if ext == "spans full canvas":
                        role = " ← primary boundary (outermost enclosing element)"
                    elif z < primary_z:
                        bg_sp = next((s for s in ctx.subpaths if s.id == rid), None)
                        area_pct = ""
                        if bg_sp and primary_area > 0:
                            ratio = bg_sp.area / primary_area * 100
                            if ratio > 100:
                                area_pct = f", area={ratio:.0f}% of primary (LARGER than primary boundary)"
                            else:
                                area_pct = f", area={ratio:.0f}% of primary"
                        shape_note = f", {shape}" if shape else ""
                        role = f" ← background layer (lower z-index, drawn first){shape_note}{area_pct}"
                    else:
                        role = ""
                    descs.append(f"{rid} ({ext}, {ch} children, {z_label}{role})")
                hints.append(
                    f"Major structural elements (top-level, ordered by draw layer — low z drawn first/behind, high z drawn last/in front): "
                    f"{'; '.join(descs)}."
                )

                # Background element layout
                bg_elements = [
                    (rid, ext, ch, z, shape)
                    for rid, ext, ch, z, shape in large_roots
                    if z < primary_z and ext != "spans full canvas"
                ]
                if bg_elements:
                    largest_bg = max(
                        bg_elements,
                        key=lambda x: next((s.area for s in ctx.subpaths if s.id == x[0]), 0),
                    )
                    lb_sp = next((s for s in ctx.subpaths if s.id == largest_bg[0]), None)
                    if lb_sp and ctx.canvas_height > 0 and ctx.canvas_width > 0:
                        lb_cy = lb_sp.centroid[1]
                        lb_y_pct = lb_cy / ctx.canvas_height * 100
                        lb_x1, lb_y1, lb_x2, lb_y2 = lb_sp.bbox
                        lb_top_pct = lb_y1 / ctx.canvas_height * 100
                        lb_bot_pct = lb_y2 / ctx.canvas_height * 100
                        lb_w_pct = (lb_x2 - lb_x1) / ctx.canvas_width * 100
                        lb_asp = lb_sp.features.get("aspect_ratio", 1.0)

                        top_sp = min(
                            (s for rid, *_ in large_roots if (s := next((sp for sp in ctx.subpaths if sp.id == rid), None))),
                            key=lambda s: s.centroid[1],
                        )
                        top_y_pct = top_sp.centroid[1] / ctx.canvas_height * 100
                        separation = abs(lb_y_pct - top_y_pct)

                        layout_parts = []
                        layout_parts.append(
                            f"Background element layout: largest background element ({largest_bg[0]}) "
                            f"centroid at y={lb_y_pct:.0f}%, bbox top edge at y={lb_top_pct:.0f}%, "
                            f"bbox bottom at y={lb_bot_pct:.0f}%"
                        )

                        if lb_top_pct < 50:
                            layout_parts.append(
                                f"This background element extends UP into the upper half of the canvas "
                                f"(top edge at y={lb_top_pct:.0f}%)"
                            )

                        if lb_w_pct > 80:
                            layout_parts.append(
                                f"Width spans {lb_w_pct:.0f}% of canvas"
                            )

                        if lb_asp > 1.3:
                            layout_parts.append(
                                f"Aspect={lb_asp:.1f} (wider than tall) — spreads laterally"
                            )
                        elif lb_asp < 0.7:
                            layout_parts.append(
                                f"Aspect={lb_asp:.1f} (taller than wide) — extends vertically"
                            )

                        if separation > 30:
                            if lb_y_pct > top_y_pct:
                                layout_parts.append(
                                    f"Top-most feature ({top_sp.id}) at y={top_y_pct:.0f}%. "
                                    f"Separation={separation:.0f}% — background element center is far below top feature"
                                )

                        hints.append(". ".join(layout_parts) + ".")

    # Cluster count
    if ctx.cluster_labels is not None:
        unique = set(int(l) for l in ctx.cluster_labels if l >= 0)
        if len(unique) > 2:
            hints.append(f"{len(unique)} spatial clusters.")

    # Shape variety
    shapes = {}
    for sp in ctx.subpaths:
        s = sp.features.get("shape_class", "organic")
        shapes[s] = shapes.get(s, 0) + 1
    if "circular" in shapes and shapes["circular"] >= 2:
        hints.append(f"{shapes['circular']} circular elements.")

    # Detail concentration — which side of the canvas has more SMALL elements
    # and concentric groups (tells the LLM where the "interesting" side is)
    if n > 15:
        canvas_mid_x = ctx.canvas_width / 2
        left_detail = 0
        right_detail = 0
        for sp in ctx.subpaths:
            tier = sp.features.get("size_tier", "MEDIUM")
            cx = sp.centroid[0]
            if tier == "SMALL":
                if cx < canvas_mid_x:
                    left_detail += 1
                else:
                    right_detail += 1
        concentric = ctx.subpaths[0].features.get("concentric_groups", []) if ctx.subpaths else []
        for group in concentric:
            center = group.get("center", (0, 0))
            members = group.get("members", [])
            if len(members) >= 2:
                if center[0] < canvas_mid_x:
                    left_detail += 2
                else:
                    right_detail += 2
        total = left_detail + right_detail
        if total > 4:
            ratio = max(left_detail, right_detail) / total
            if ratio > 0.65:
                side = "right" if right_detail > left_detail else "left"
                hints.append(f"Detail/features concentrated on {side} side of canvas.")

    # Grid shape
    if grid:
        col_profile = _column_profile(grid)
        row_profile = _row_profile(grid)
        filled_cols = [i for i, v in enumerate(col_profile) if v > 15]
        if filled_cols:
            left_edge = filled_cols[0]
            right_edge = filled_cols[-1]
            span = right_edge - left_edge + 1
            total_cols = len(col_profile)
            if span < total_cols * 0.7:
                side = "left" if left_edge < total_cols * 0.2 else ("right" if right_edge > total_cols * 0.8 else "center")
                hints.append(f"Silhouette spans {span}/{total_cols} columns, toward {side}.")

    return hints


def _decompose_clusters(ctx: PipelineContext) -> list[str]:
    """Decompose the scene into cluster-based visual units."""
    if ctx.cluster_labels is None:
        return []

    labels = ctx.cluster_labels
    unique = sorted(set(int(l) for l in labels if l >= 0))
    if len(unique) < 2:
        return []

    canvas_w, canvas_h = ctx.canvas_width, ctx.canvas_height
    descriptions = []

    cluster_info = []
    for label in unique:
        indices = [i for i in range(len(labels)) if labels[i] == label]
        if len(indices) < 3:
            continue

        members = [ctx.subpaths[i] for i in indices]
        centroids_x = [sp.centroid[0] for sp in members]
        centroids_y = [sp.centroid[1] for sp in members]
        avg_x = sum(centroids_x) / len(centroids_x)
        avg_y = sum(centroids_y) / len(centroids_y)

        # Quadrant
        h_pos = "left" if avg_x < canvas_w * 0.33 else ("center" if avg_x < canvas_w * 0.66 else "right")
        v_pos = "upper" if avg_y < canvas_h * 0.33 else ("mid" if avg_y < canvas_h * 0.66 else "lower")

        # Dominant shape type
        shape_counts: dict[str, int] = {}
        for sp in members:
            s = sp.features.get("shape_class", "organic")
            shape_counts[s] = shape_counts.get(s, 0) + 1
        dominant_shape = max(shape_counts, key=shape_counts.get) if shape_counts else "mixed"

        # Total area
        total_area = sum(sp.area for sp in members)

        cluster_info.append({
            "label": label,
            "count": len(indices),
            "pos": f"{v_pos}-{h_pos}",
            "cx": avg_x,
            "cy": avg_y,
            "shape": dominant_shape,
            "area": total_area,
        })

        descriptions.append(
            f"Cluster {label} ({v_pos}-{h_pos}, {len(indices)} elements, "
            f"dominant shape: {dominant_shape})"
        )

    # Inter-cluster relationships
    if len(cluster_info) >= 2:
        for i in range(len(cluster_info)):
            for j in range(i + 1, len(cluster_info)):
                ci = cluster_info[i]
                cj = cluster_info[j]
                dx = cj["cx"] - ci["cx"]
                dy = cj["cy"] - ci["cy"]
                dist = (dx**2 + dy**2) ** 0.5
                descriptions.append(
                    f"Cluster {ci['label']} ({ci['pos']}) — "
                    f"Cluster {cj['label']} ({cj['pos']}) — "
                    f"separated by {dist:.0f}px"
                )

    return descriptions

"""PipelineContext → enrichment text and EnrichmentOutput model.

Structure optimized for LLM spatial reasoning (v12):
  Interpretation → Shape Narrative → Visual Pyramid → Scene Walkthrough → Structure
"""

from __future__ import annotations

import math

import numpy as np

from app.engine.context import PipelineContext, SubPathData
from app.svg.anonymizer import sanitize_tag
from app.models.enrichment import (
    ClusterInfo,
    ComponentInfo,
    ContainmentRelation,
    ElementSummary,
    EnrichmentOutput,
    SilhouetteInfo,
    SymmetryInfo,
)


# ── Importance scoring weights ──
# Multi-criteria visual salience: area is 2× any individual feature.
# Total range: [-5, 95].  Budget: area(30) + circ(15) + depth(15)
#   + contrast(15) + rarity(10) + concentric(10) - isolation(5) = 90.
_IMP_AREA_MAX = 30.0          # Area: largest weight (2×features)
_IMP_AREA_SCALE = 3.0         # area_pct × 100 × 3 → max 30 at 10% of total
_IMP_CIRC_WEIGHT = 15.0       # Circularity: 0-15 (= _IMP_AREA_MAX / 2)
_IMP_DEPTH_MAX = 15.0         # Containment depth cap
_IMP_DEPTH_SCALE = 5.0        # 3 levels × 5 = 15 (caps at max)
_IMP_CONTRAST_WEIGHT = 15.0   # Color contrast: 0-15
_IMP_RARITY_WEIGHT = 10.0     # Shape uniqueness: 0-10
_IMP_CONCENTRIC_BONUS = 10.0  # Concentric group membership
_IMP_ISOLATION_PENALTY = 5.0  # Small isolated element
_IMP_CIRC_THRESHOLD = 0.50    # 1/2 — below = not circular enough to avoid penalty

# ── Position grid — Rule of Thirds ──
_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0

# ── Visual pyramid grid resolutions ──
# Token budget: each level uses ~50-100 words of ASCII. Total ~300 words.
_L0_RESOLUTION = 12   # Coarsest: overview of all elements
_L2_RESOLUTION = 14   # Interior features within boundary

# ── Visual pyramid fill thresholds ──
_L0_MIN_FILLED = 3    # Need ≥3 cells filled to render L0
_L0_MAX_FILL = 0.85   # >85% filled = nearly rectangular, skip L0 (≈5/6)
_L2_MIN_FILLED = 3    # Need ≥3 cells for interior

# Interior area: elements < 90% of primary boundary area → not the boundary itself.
_INTERIOR_AREA_RATIO = 0.9  # 9/10

# Edge exclusion: groups within 2% of canvas edge are decorative.
_EDGE_EXCLUSION_PCT = 2.0  # 2% ≈ design-grid half-margin

# Maximum concentric group zooms to render
_MAX_GROUP_ZOOMS = 6

# Cumulative coverage: show children until 90% of parent area covered.
# Pareto-inspired: 90% coverage captures nearly all visual weight.
_CUMULATIVE_COVERAGE = 0.90  # 9/10

# Minimum elements for complexity to be worth importance scoring
_COMPLEXITY_MIN_ELEMENTS = 8

# ── Path simplification (RDP) ──
# 5% of max bbox dimension → typically 5-12 vertices per element.
_RDP_EPSILON_FRAC = 0.05
# Hard cap on output vertices to keep token budget bounded.
_RDP_MAX_VERTICES = 12
# Need at least 4 sampled points to produce a meaningful simplification.
_RDP_MIN_POINTS = 4


def _entropy_cap(values: list[float], safety_max: int) -> int:
    """Shannon entropy perplexity: effective number of equally-important items.

    H = -sum(p_i * log2(p_i))  (Shannon, 1948)
    Perplexity = ceil(2^H) = effective count of uniform-equivalent items.

    Reuses pattern from pixel_segmentation._effective_region_count().
    """
    if not values or all(v <= 0 for v in values):
        return min(len(values), safety_max)
    arr = np.array([v for v in values if v > 0], dtype=float)
    if len(arr) <= 1:
        return len(arr)
    total = arr.sum()
    if total <= 0:
        return len(arr)
    p = arr / total
    H = -np.sum(p * np.log2(p + np.finfo(float).eps))
    perplexity = int(np.ceil(2.0 ** H))
    return min(max(perplexity, 1), safety_max)


def _coverage_cap(
    children_areas: list[float], total_area: float, safety_max: int,
) -> int:
    """Cumulative area coverage: show children until >=90% of parent covered.

    Returns the number of children needed (sorted by area descending).
    """
    if total_area <= 0:
        return min(len(children_areas), safety_max)
    sorted_areas = sorted(children_areas, reverse=True)
    cumulative = 0.0
    for i, a in enumerate(sorted_areas):
        cumulative += a
        if cumulative / total_area >= _CUMULATIVE_COVERAGE:
            return min(i + 1, safety_max)
    return min(len(sorted_areas), safety_max)


def _compute_importance(sp: SubPathData, ctx: PipelineContext) -> float:
    """Score element visual salience (0-100). Higher = more important to show."""
    score = 0.0
    n = len(ctx.subpaths)
    if n == 0:
        return 0.0

    # Area contribution (0-30): larger elements are more visually dominant
    total_area = sum(s.area for s in ctx.subpaths) or 1.0
    area_pct = sp.area / total_area
    score += min(_IMP_AREA_MAX, area_pct * 100 * _IMP_AREA_SCALE)

    # Circularity bonus (0-15): circular elements are focal (eyes, buttons, dots)
    circ = sp.features.get("circularity", 0.0)
    score += circ * _IMP_CIRC_WEIGHT

    # Containment depth (0-15): elements nested deeply are interesting features
    if ctx.containment_matrix is not None:
        idx = next((i for i, s in enumerate(ctx.subpaths) if s.id == sp.id), -1)
        if idx >= 0:
            depth = sum(1 for j in range(n) if j != idx and ctx.containment_matrix[j][idx])
            score += min(_IMP_DEPTH_MAX, depth * _IMP_DEPTH_SCALE)

    # Color contrast (0-15): elements with high contrast against parent are salient
    contrasts = sp.features.get("color_contrasts", [])
    if contrasts:
        max_contrast = max(c[1] for c in contrasts)
        score += max_contrast * _IMP_CONTRAST_WEIGHT

    # Shape uniqueness (0-10): rare shapes within the SVG stand out
    shape_class = sp.features.get("shape_class", "organic")
    shape_counts: dict[str, int] = {}
    for s in ctx.subpaths:
        sc = s.features.get("shape_class", "organic")
        shape_counts[sc] = shape_counts.get(sc, 0) + 1
    rarity = 1.0 - (shape_counts.get(shape_class, 1) / n)
    score += rarity * _IMP_RARITY_WEIGHT

    # Concentric membership bonus (0-10): part of a concentric group = focal
    concentric = ctx.subpaths[0].features.get("concentric_groups", []) if ctx.subpaths else []
    for group in concentric:
        if sp.id in group.get("members", []):
            score += _IMP_CONCENTRIC_BONUS
            break

    # Small isolated element penalty (-5): tiny scattered elements are less important
    tier = sp.features.get("size_tier", "MEDIUM")
    if tier == "SMALL" and circ < _IMP_CIRC_THRESHOLD:
        score -= _IMP_ISOLATION_PENALTY

    return max(0.0, score)


def context_to_enrichment(ctx: PipelineContext) -> EnrichmentOutput:
    """Convert PipelineContext to structured EnrichmentOutput."""
    elements = []
    for sp in ctx.subpaths:
        elements.append(
            ElementSummary(
                id=sp.id,
                shape_class=sp.features.get("shape_class", "organic"),
                area=sp.area,
                bbox=sp.bbox,
                centroid=sp.centroid,
                circularity=sp.features.get("circularity", 0.0),
                convexity=sp.features.get("convexity", 0.0),
                aspect_ratio=sp.features.get("aspect_ratio", 1.0),
                size_tier=sp.features.get("size_tier", "MEDIUM"),
            )
        )

    enrichment_text = context_to_enrichment_text(ctx)

    return EnrichmentOutput(
        source="uploaded",
        canvas=(ctx.canvas_width, ctx.canvas_height),
        element_count=len(ctx.subpaths),
        subpath_count=len(ctx.subpaths),
        is_stroke_based=ctx.is_stroke_based,
        elements=elements,
        symmetry=SymmetryInfo(
            axis_type=ctx.symmetry_axis or "none",
            score=ctx.symmetry_score,
            pairs=[(ctx.subpaths[a].id, ctx.subpaths[b].id) for a, b in ctx.symmetry_pairs if a < len(ctx.subpaths) and b < len(ctx.subpaths)],
        ),
        ascii_grid_positive=ctx.ascii_grid_positive,
        ascii_grid_negative=ctx.ascii_grid_negative,
        enrichment_text=enrichment_text,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helper: canvas position label
# ═══════════════════════════════════════════════════════════════════════════


def _region_label(cx: float, cy: float, cw: float, ch: float) -> str:
    """Return canvas position label like 'upper-right'."""
    px = cx / cw if cw > 0 else 0.5
    py = cy / ch if ch > 0 else 0.5
    v = "top" if py < _THIRD else ("mid" if py < _TWO_THIRDS else "bottom")
    h = "left" if px < _THIRD else ("center" if px < _TWO_THIRDS else "right")
    if v == "mid" and h == "center":
        return "center"
    if v == "mid":
        return h
    if h == "center":
        return v
    return f"{v}-{h}"


# ═══════════════════════════════════════════════════════════════════════════
# Helper: ASCII grid renderer
# ═══════════════════════════════════════════════════════════════════════════


def _render_ascii_grid(grid: np.ndarray) -> str:
    """Render a 2D numpy binary grid as #/. ASCII text."""
    rows, cols = grid.shape
    lines = []
    for r in range(rows):
        lines.append("".join("#" if grid[r, c] else "." for c in range(cols)))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Simplified path (RDP)
# ═══════════════════════════════════════════════════════════════════════════


def _simplify_path(sp: SubPathData) -> str:
    """Simplify element outline to ~5-12 vertices, normalized to bbox 0-100.

    Uses the polygon exterior (actual shape boundary) rather than raw SVG
    draw-path points, so the result is the outline silhouette — no interior
    zigzags.  Falls back to LineString simplification for degenerate shapes.

    Returns a compact SVG-like path string, e.g.:
        path(8v): M 5,50 L 15,10 L 50,0 L 85,10 L 95,50 L 85,90 Z
    """
    from shapely.geometry import LineString, MultiPolygon, Polygon

    pts = sp.points
    if len(pts) < _RDP_MIN_POINTS:
        return ""

    pts_arr = np.asarray(pts, dtype=float)
    if pts_arr.ndim != 2 or pts_arr.shape[1] < 2:
        return ""

    x1, y1, x2, y2 = sp.bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 and h <= 0:
        return ""

    max_dim = max(w, h)
    epsilon = max_dim * _RDP_EPSILON_FRAC

    # Try polygon exterior first (gives the actual outline, always closed)
    is_closed = False
    try:
        poly = Polygon(pts_arr[:, :2])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            raise ValueError("degenerate polygon")
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        exterior = poly.exterior.simplify(epsilon, preserve_topology=True)
        coords = list(exterior.coords)
        # Exterior rings repeat first point as last — remove it
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        is_closed = True
    except Exception:
        # Fallback: LineString for degenerate/open shapes
        try:
            line = LineString(pts_arr[:, :2])
            simplified = line.simplify(epsilon, preserve_topology=True)
            coords = list(simplified.coords)
        except Exception:
            return ""

    # Increase epsilon if too many vertices
    attempts = 0
    while len(coords) > _RDP_MAX_VERTICES and attempts < 5:
        epsilon *= 1.5
        try:
            if is_closed:
                poly2 = Polygon(pts_arr[:, :2])
                if not poly2.is_valid:
                    poly2 = poly2.buffer(0)
                if isinstance(poly2, MultiPolygon):
                    poly2 = max(poly2.geoms, key=lambda g: g.area)
                ext2 = poly2.exterior.simplify(epsilon, preserve_topology=True)
                coords = list(ext2.coords)
                if len(coords) > 1 and coords[0] == coords[-1]:
                    coords = coords[:-1]
            else:
                line2 = LineString(pts_arr[:, :2])
                coords = list(line2.simplify(epsilon, preserve_topology=True).coords)
        except Exception:
            break
        attempts += 1
    if len(coords) > _RDP_MAX_VERTICES:
        coords = coords[:_RDP_MAX_VERTICES]

    if len(coords) < 3:
        return ""

    # Normalize to element bounding box (0-100%) + deduplicate consecutive
    safe_w = w if w > 0 else 1.0
    safe_h = h if h > 0 else 1.0
    norm: list[tuple[int, int]] = []
    for x, y in coords:
        nx = max(0, min(100, int(round((x - x1) / safe_w * 100))))
        ny = max(0, min(100, int(round((y - y1) / safe_h * 100))))
        if norm and norm[-1] == (nx, ny):
            continue  # Deduplicate consecutive identical vertices
        norm.append((nx, ny))

    if len(norm) < 3:
        return ""

    # Build SVG-like path string
    parts = [f"M {norm[0][0]},{norm[0][1]}"]
    for px, py in norm[1:]:
        parts.append(f"L {px},{py}")
    if is_closed:
        parts.append("Z")

    return f"path({len(norm)}v): {' '.join(parts)}"


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Visual Pyramid
# ═══════════════════════════════════════════════════════════════════════════


def _render_visual_pyramid(
    ctx: PipelineContext,
    primary_boundary,
) -> list[str]:
    """Spatial overview grids: composite silhouette + interior feature map."""
    from app.utils.rasterizer import multi_element_grid

    lines = ["\u2500\u2500 VISUAL PYRAMID \u2500\u2500", ""]
    cw, ch = ctx.canvas_width, ctx.canvas_height

    # Level 0 — Full composite silhouette (12×12)
    if ctx.subpaths:
        point_arrays = [
            np.asarray(sp.points) for sp in ctx.subpaths if len(sp.points) > 0
        ]
        if point_arrays:
            grid_l0 = multi_element_grid(point_arrays, cw, ch, resolution=_L0_RESOLUTION)
            fill_pct = np.sum(grid_l0) / grid_l0.size
            if np.sum(grid_l0) >= _L0_MIN_FILLED and fill_pct < _L0_MAX_FILL:
                lines.append(f"ALL ELEMENTS ({_L0_RESOLUTION}\u00d7{_L0_RESOLUTION}, #=filled .=empty):")
                lines.append(_render_ascii_grid(grid_l0))
                lines.append("")

    # Interior features only (14×14)
    # Everything INSIDE the primary boundary, WITHOUT the boundary itself.
    if (
        primary_boundary is not None
        and ctx.containment_matrix is not None
    ):
        primary_idx = next(
            (i for i, sp in enumerate(ctx.subpaths)
             if sp.id == primary_boundary.id),
            -1,
        )
        if primary_idx >= 0:
            interior_sps = [
                ctx.subpaths[j]
                for j in range(len(ctx.subpaths))
                if j != primary_idx
                and ctx.containment_matrix[primary_idx][j]
                and ctx.subpaths[j].area < primary_boundary.area * _INTERIOR_AREA_RATIO
            ]
            if interior_sps:
                interior_points = [
                    np.asarray(sp.points)
                    for sp in interior_sps
                    if len(sp.points) > 0
                ]
                if interior_points:
                    grid_l2 = multi_element_grid(
                        interior_points, cw, ch, resolution=_L2_RESOLUTION,
                    )
                    if np.sum(grid_l2) >= _L2_MIN_FILLED:
                        lines.append(
                            f"INTERIOR FEATURES "
                            f"({_L2_RESOLUTION}\u00d7{_L2_RESOLUTION}, {len(interior_sps)} elements inside "
                            f"{primary_boundary.id}, boundary removed):"
                        )
                        lines.append(_render_ascii_grid(grid_l2))
                        lines.append("")

    return lines


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Scene Walkthrough
# ═══════════════════════════════════════════════════════════════════════════


def _build_scene_walkthrough(
    ctx: PipelineContext,
    sorted_sps: list[SubPathData],
    primary_boundary,
    concentric: list[dict],
    is_complex: bool,
    importance_scores: dict[str, float],
    top_n: int,
) -> list[str]:
    """Scene walkthrough: co-locate all element data by spatial hierarchy.

    Replaces 5 separate sections (Reconstruction, Key Elements, Spatial
    Connections, Containment Tree, Key Overlaps) with a single outside-to-inside
    walk that puts shape, metrics, color, children, touches, and overlaps
    together for each element.
    """
    lines = ["\u2500\u2500 SCENE WALKTHROUGH (outside \u2192 inside) \u2500\u2500", ""]

    if not ctx.subpaths:
        return lines

    n = len(ctx.subpaths)
    cw, ch = ctx.canvas_width, ctx.canvas_height

    # Build index lookups
    id_to_idx: dict[str, int] = {sp.id: i for i, sp in enumerate(ctx.subpaths)}

    # Build containment hierarchy (direct parent + children)
    parent: dict[int, int | None] = {i: None for i in range(n)}
    children_map: dict[int, list[int]] = {i: [] for i in range(n)}

    if ctx.containment_matrix is not None:
        for j in range(n):
            candidates = [
                i for i in range(n)
                if i != j and ctx.containment_matrix[i][j]
            ]
            if candidates:
                parent[j] = min(candidates, key=lambda i: ctx.subpaths[i].area)
        for j in range(n):
            if parent[j] is not None:
                children_map[parent[j]].append(j)

    # Collect touching pairs — filter ancestor-descendant pairs.
    # Containment matrix is transitive (geometric contains), so
    # cmat[A][B] means A is an ancestor of B.
    touching_pairs: list[tuple[str, str, float]] = []
    if ctx.subpaths:
        raw_touching = ctx.subpaths[0].features.get("touching_pairs", [])
        for id_a, id_b, dist in raw_touching:
            idx_a = id_to_idx.get(id_a, -1)
            idx_b = id_to_idx.get(id_b, -1)
            if idx_a < 0 or idx_b < 0:
                continue
            if ctx.containment_matrix is not None:
                if ctx.containment_matrix[idx_a][idx_b] or ctx.containment_matrix[idx_b][idx_a]:
                    continue
            touching_pairs.append((id_a, id_b, dist))

    # Collect all overlaps (deduplicated)
    all_overlaps: list[tuple[str, str, float]] = []
    seen_ov: set[tuple[str, str]] = set()
    for sp in ctx.subpaths:
        for ov in sp.features.get("overlaps", []):
            pair = tuple(sorted([sp.id, ov["element"]]))
            iou = ov.get("iou", 0)
            if iou > 0.05 and pair not in seen_ov:
                all_overlaps.append((pair[0], pair[1], iou))
                seen_ov.add(pair)

    # ── Per-element inline formatter ──
    def _fmt_element(sp: SubPathData, indent: str = "  ") -> str:
        shape_class = sp.features.get("shape_class", "organic")
        turning = sp.features.get("turning_classification", "")
        circ = sp.features.get("circularity", 0.0)
        conv = sp.features.get("convexity", 0.0)
        aspect = sp.features.get("aspect_ratio", 1.0)
        tier = sp.features.get("size_tier", "MEDIUM")
        color_label = sp.features.get("color_label", "")
        fill_color = sp.features.get("fill_color", "")
        x1, y1, x2, y2 = sp.bbox
        cov_w = (x2 - x1) / cw * 100 if cw > 0 else 0
        cov_h = (y2 - y1) / ch * 100 if ch > 0 else 0

        parts = [f"{indent}{sp.id} [{tier}] {shape_class}"]
        if turning:
            parts[0] += f"({turning})"
        parts.append(f"covers {cov_w:.0f}%\u00d7{cov_h:.0f}%")
        parts.append(f"circ={circ:.2f}, conv={conv:.2f}, aspect={aspect:.2f}")
        if color_label and color_label != "none":
            color_part = f"color={color_label}"
            if fill_color:
                color_part += f"({fill_color})"
            parts.append(color_part)
        if is_complex and sp.id in importance_scores:
            parts.append(f"salience={importance_scores[sp.id]:.0f}")
        return ", ".join(parts)

    # ── Per-element relation lines (children, touches, overlaps) ──
    def _fmt_relations(sp_id: str, indent: str = "    ") -> list[str]:
        rel_lines: list[str] = []
        idx = id_to_idx.get(sp_id, -1)

        # Children
        if idx >= 0 and children_map[idx]:
            child_ids = [ctx.subpaths[ci].id for ci in children_map[idx]]
            child_areas = [ctx.subpaths[ci].area for ci in children_map[idx]]
            total = ctx.subpaths[idx].area if ctx.subpaths[idx].area > 0 else sum(child_areas)
            cap = _coverage_cap(child_areas, total, 10)
            shown = ", ".join(child_ids[:cap])
            extra = f" +{len(child_ids) - cap}" if len(child_ids) > cap else ""
            rel_lines.append(f"{indent}\u2192 children: {shown}{extra}")

        # Touches (only pairs involving this element, already ancestor-filtered)
        elem_touches = [
            (a, b, d) for a, b, d in touching_pairs
            if a == sp_id or b == sp_id
        ]
        if elem_touches:
            touch_strs = []
            for a, b, d in elem_touches[:5]:
                other = b if a == sp_id else a
                touch_strs.append(f"{other}({d:.1f}px)")
            rel_lines.append(f"{indent}\u2192 touches: {', '.join(touch_strs)}")

        # Overlaps involving this element
        elem_overlaps = [
            (a, b, iou) for a, b, iou in all_overlaps
            if a == sp_id or b == sp_id
        ]
        if elem_overlaps:
            ov_strs = []
            for a, b, iou in elem_overlaps[:5]:
                ov_strs.append(f"{a}\u2194{b}(IoU={iou:.3f})")
            rel_lines.append(f"{indent}\u2192 overlaps: {', '.join(ov_strs)}")

        return rel_lines

    described: set[str] = set()

    def _emit_path(sp: SubPathData, indent: str) -> None:
        """Append simplified path line for the element."""
        path = _simplify_path(sp)
        if path:
            lines.append(f"{indent}{path}")

    # ═══════════════════════════════════════════════════════════
    # Complex SVGs: primary boundary + containment hierarchy
    # ═══════════════════════════════════════════════════════════
    if primary_boundary is not None and ctx.containment_matrix is not None:
        primary_idx = id_to_idx.get(primary_boundary.id, -1)

        if primary_idx >= 0:
            # 1. Background: only elements >= 90% of primary area AND
            #    not contained by primary. These are true canvas-filling
            #    backgrounds (e.g. body fill behind the outline).
            bg_elements = []
            for i, sp in enumerate(ctx.subpaths):
                if sp.id == primary_boundary.id:
                    continue
                if sp.area >= primary_boundary.area * _INTERIOR_AREA_RATIO:
                    if not ctx.containment_matrix[primary_idx][i]:
                        bg_elements.append(sp)

            if bg_elements:
                lines.append("Background:")
                for sp in bg_elements:
                    lines.append(_fmt_element(sp, "  "))
                    _emit_path(sp, "    ")
                    lines.extend(_fmt_relations(sp.id, "    "))
                    described.add(sp.id)
                lines.append("")

            # 2. Primary boundary (the figure outline)
            lines.append("Primary boundary:")
            lines.append(_fmt_element(primary_boundary, "  "))
            _emit_path(primary_boundary, "    ")
            lines.extend(_fmt_relations(primary_boundary.id, "    "))
            described.add(primary_boundary.id)
            lines.append("")

            # 3. Feature groups (concentric) — all groups regardless of
            #    strict containment, since complex outlines often have
            #    features that overlap the boundary rather than sitting
            #    fully inside it.
            if concentric:
                _edge_frac = _EDGE_EXCLUSION_PCT / 100
                feature_groups = []
                for group in concentric:
                    members = group.get("members", [])
                    center = group.get("center", (0, 0))
                    if len(members) < 2:
                        continue
                    if (center[0] < cw * _edge_frac
                            or center[0] > cw * (1 - _edge_frac)
                            or center[1] < ch * _edge_frac
                            or center[1] > ch * (1 - _edge_frac)):
                        continue
                    # Skip groups where ALL members already described
                    undescribed = [m for m in members if m not in described]
                    if not undescribed:
                        continue
                    feature_groups.append(group)

                feature_groups.sort(
                    key=lambda g: len(g.get("members", [])), reverse=True,
                )
                if feature_groups:
                    _grp_sizes = [
                        float(len(g.get("members", [])))
                        for g in feature_groups
                    ]
                    _grp_cap = _entropy_cap(_grp_sizes, _MAX_GROUP_ZOOMS)

                    for group in feature_groups[:_grp_cap]:
                        members = group.get("members", [])
                        center = group.get("center", (0, 0))
                        pos = _region_label(center[0], center[1], cw, ch)
                        member_sps = [
                            sp for sp in ctx.subpaths if sp.id in members
                        ]
                        max_circ = max(
                            (sp.features.get("circularity", 0)
                             for sp in member_sps),
                            default=0,
                        )
                        member_str = "+".join(members)
                        lines.append(
                            f"Feature group [{member_str}] at {pos} "
                            f"({len(members)} concentric layers, "
                            f"max circ={max_circ:.2f}):"
                        )
                        for sp in member_sps:
                            if sp.id in described:
                                continue
                            lines.append(_fmt_element(sp, "    "))
                            _emit_path(sp, "      ")
                            lines.extend(_fmt_relations(sp.id, "      "))
                            described.add(sp.id)
                        lines.append("")

            # 4. Remaining individuals (LARGE/MEDIUM not yet described).
            #    No containment requirement — elements may overlap the
            #    boundary rather than sit strictly inside it.
            remaining = []
            for sp in sorted_sps:
                if sp.id in described:
                    continue
                if sp.features.get("size_tier") not in ("LARGE", "MEDIUM"):
                    continue
                remaining.append(sp)

            if remaining:
                _ri_areas = [sp.area for sp in remaining]
                _ri_cap = _entropy_cap(_ri_areas, 12)
                for sp in remaining[:_ri_cap]:
                    sp_idx = id_to_idx.get(sp.id, -1)
                    inside = (
                        sp_idx >= 0
                        and ctx.containment_matrix[primary_idx][sp_idx]
                    )
                    loc = (
                        f"Inside {primary_boundary.id}"
                        if inside else "Feature"
                    )
                    lines.append(f"{loc}, individual:")
                    lines.append(_fmt_element(sp, "    "))
                    _emit_path(sp, "      ")
                    lines.extend(_fmt_relations(sp.id, "      "))
                    described.add(sp.id)
                    lines.append("")

            # 5. Small detail summary — all non-described elements
            undescribed_small = [
                sp for sp in ctx.subpaths
                if sp.id not in described
                and sp.id != primary_boundary.id
            ]
            if undescribed_small:
                lines.append(
                    f"+ {len(undescribed_small)} small detail elements"
                )
                lines.append("")

    # ═══════════════════════════════════════════════════════════
    # Simple SVGs: no primary boundary, list by area
    # ═══════════════════════════════════════════════════════════
    else:
        _simple_areas = [sp.area for sp in sorted_sps]
        _simple_cap = _entropy_cap(_simple_areas, top_n)
        for sp in sorted_sps[:_simple_cap]:
            lines.append(_fmt_element(sp, "  "))
            _emit_path(sp, "    ")
            lines.extend(_fmt_relations(sp.id, "    "))
            if sp.source_tag:
                lines.append(f"    svg: {sanitize_tag(sp.source_tag)}")
            described.add(sp.id)
        remaining = len(sorted_sps) - min(_simple_cap, len(sorted_sps))
        if remaining > 0:
            lines.append(f"+ {remaining} additional smaller elements")
        lines.append("")

    return lines


# ═══════════════════════════════════════════════════════════════════════════
# Main enrichment text generator
# ═══════════════════════════════════════════════════════════════════════════


def context_to_enrichment_text(ctx: PipelineContext) -> str:
    """Format PipelineContext as enrichment text block for LLM injection.

    Structure optimized for LLM spatial reasoning (v12):
      1. Spatial Interpretation (highest-signal synthesis)
      2. Shape Narrative (contour walk, feature pairs, structural detail)
      3. Visual Pyramid (multi-level ASCII zooms + primary boundary Braille)
      4. Scene Walkthrough (co-located element data by spatial hierarchy)
      5. Structure (symmetry, colors, repeated elements)
      6. Learned Patterns

    Key insight: co-locate all properties of each element (shape, metrics,
    color, containment children, touching neighbors, overlaps) in a single
    walk from outside to inside, so the LLM never has to mentally grep
    across sections.
    """
    from app.engine.interpreter import interpret, _find_primary_boundary

    lines = ["=== VECTORSIGHT ENRICHMENT ===", ""]
    n_elements = len(ctx.subpaths)
    lines.append(
        f"ELEMENTS: {n_elements} paths | "
        f"CANVAS: {ctx.canvas_width:.0f}\u00d7{ctx.canvas_height:.0f} | "
        f"TYPE: {'stroke-based' if ctx.is_stroke_based else 'fill-based'}"
    )
    lines.append("")

    # Complexity detection: area distribution entropy determines if importance
    # scoring (expensive) is worth it. Complex when perplexity > sqrt(n) AND
    # n >= minimum threshold for scoring to differentiate elements.
    areas = [sp.area for sp in ctx.subpaths]
    area_perplexity = _entropy_cap(areas, n_elements)
    is_complex = (
        n_elements >= _COMPLEXITY_MIN_ELEMENTS
        and area_perplexity > math.isqrt(max(n_elements, 1))
    )
    if is_complex:
        scored = [(sp, _compute_importance(sp, ctx)) for sp in ctx.subpaths]
        scored.sort(key=lambda x: x[1], reverse=True)
        sorted_sps = [sp for sp, _ in scored]
        importance_scores = {sp.id: sc for sp, sc in scored}
        # Entropy perplexity of importance scores determines how many to show
        imp_values = [sc for _, sc in scored if sc > 0]
        top_n = _entropy_cap(imp_values, 15)
    else:
        sorted_sps = sorted(ctx.subpaths, key=lambda sp: sp.area, reverse=True)
        importance_scores = {}
        # Entropy perplexity of area distribution determines how many to show
        top_n = _entropy_cap(areas, 18)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: SPATIAL INTERPRETATION (highest-value signal, first)
    # ═══════════════════════════════════════════════════════════════════

    interp = interpret(ctx)
    interp_text = interp.to_text()
    if interp_text:
        lines.append(interp_text)
        lines.append("")

    # ── Shape Narrative (contour walk + feature synthesis) ──
    if interp.shape_narrative:
        narrative_text = interp.shape_narrative.to_text()
        if narrative_text:
            lines.append(narrative_text)
            lines.append("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: VISUAL PYRAMID (multi-level ASCII zooms)
    # ═══════════════════════════════════════════════════════════════════

    primary_boundary = _find_primary_boundary(ctx)
    concentric: list[dict] = []
    if ctx.subpaths:
        concentric = ctx.subpaths[0].features.get("concentric_groups", [])

    lines.extend(
        _render_visual_pyramid(ctx, primary_boundary)
    )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: SCENE WALKTHROUGH (co-located element data)
    # ═══════════════════════════════════════════════════════════════════

    lines.extend(
        _build_scene_walkthrough(
            ctx, sorted_sps, primary_boundary, concentric,
            is_complex, importance_scores, top_n,
        )
    )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: STRUCTURE
    # ═══════════════════════════════════════════════════════════════════

    # ── Symmetry (T1.20) ──
    if ctx.symmetry_axis:
        lines.append(
            f"SYMMETRY: {ctx.symmetry_axis} "
            f"(score {ctx.symmetry_score:.2f})"
        )
        if ctx.symmetry_pairs:
            pair_strs = []
            # All detected symmetry pairs are meaningful; cap at 10 for readability
            for a, b in ctx.symmetry_pairs[:10]:
                if a >= n_elements or b >= n_elements:
                    continue
                sp_a, sp_b = ctx.subpaths[a], ctx.subpaths[b]
                pair_strs.append(f"({sp_a.id}\u2194{sp_b.id})")
            if pair_strs:
                lines.append(f"  Pairs: {', '.join(pair_strs)}")
        lines.append("")

    # ── Color palette (T1.25) ──
    if ctx.subpaths:
        palette = ctx.subpaths[0].features.get("palette", [])
        color_groups = ctx.subpaths[0].features.get("color_groups", {})
        if palette:
            _color_freqs = [float(count) for _, count in palette]
            _palette_cap = _entropy_cap(_color_freqs, 8)
            palette_strs = [
                f"{color} \u00d7{count}" for color, count in palette[:_palette_cap]
            ]
            lines.append(f"COLOR PALETTE: {', '.join(palette_strs)}")
            if color_groups and len(color_groups) > 1:
                _grp_sizes = [float(len(m)) for _, m in sorted(
                    color_groups.items(), key=lambda x: len(x[1]), reverse=True,
                )]
                _grp_cap = _entropy_cap(_grp_sizes, 6)
                for label, members in sorted(
                    color_groups.items(),
                    key=lambda x: len(x[1]),
                    reverse=True,
                )[:_grp_cap]:
                    if len(members) >= 2:
                        # Show members until 90% cumulative coverage
                        _mem_areas = [
                            next((sp.area for sp in ctx.subpaths if sp.id == m), 1.0)
                            for m in members
                        ]
                        _mem_total = sum(_mem_areas)
                        _mem_cap = _coverage_cap(_mem_areas, _mem_total, 10)
                        ids = ', '.join(members[:_mem_cap])
                        extra = (
                            f" +{len(members) - _mem_cap} more"
                            if len(members) > _mem_cap else ""
                        )
                        lines.append(f"  {label}: [{ids}]{extra}")
            # High-contrast pairs
            contrast_pairs: list[tuple[str, str, float]] = []
            for sp in ctx.subpaths:
                for child_id, contrast in sp.features.get(
                    "color_contrasts", []
                ):
                    if contrast >= 0.4:
                        contrast_pairs.append((sp.id, child_id, contrast))
            if contrast_pairs:
                contrast_pairs.sort(key=lambda x: x[2], reverse=True)
                _cp_values = [float(v) for _, _, v in contrast_pairs]
                _cp_cap = _entropy_cap(_cp_values, 8)
                lines.append(
                    f"  High contrast: "
                    f"{', '.join(f'{p}\u2192{c}({v:.1f})' for p, c, v in contrast_pairs[:_cp_cap])}"
                )
            lines.append("")

    # ── Repeated elements (T3.09 + T3.14, compact) ──
    rep_groups = []
    if ctx.subpaths:
        rep_groups = ctx.subpaths[0].features.get("repetition_groups", [])
    if rep_groups:
        total = len(ctx.subpaths)
        rep_groups = [
            g for g in rep_groups
            if g.get("count", len(g.get("members", []))) / total <= 0.8
        ]
    if rep_groups:
        _rg_counts = [float(g.get("count", len(g.get("members", [])))) for g in rep_groups]
        _rg_cap = _entropy_cap(_rg_counts, 6)
        lines.append("REPEATED ELEMENTS:")
        for group in rep_groups[:_rg_cap]:
            members = group.get("members", [])
            count = group.get("count", len(members))
            shape = group.get("shape_class", "unknown")
            pattern = group.get("pattern", "")
            # Show members until 90% cumulative coverage
            _rm_areas = [
                next((sp.area for sp in ctx.subpaths if sp.id == m), 1.0)
                for m in members
            ]
            _rm_total = sum(_rm_areas)
            _rm_cap = _coverage_cap(_rm_areas, _rm_total, 8)
            line = f"  [{', '.join(members[:_rm_cap])}] \u00d7{count} {shape}"
            if pattern and pattern != "none":
                line += f", pattern={pattern}"
            lines.append(line)
        lines.append("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: LEARNED PATTERNS
    # ═══════════════════════════════════════════════════════════════════

    try:
        from app.learning.memory import get_memory_store

        store = get_memory_store()
        shape_dist: dict[str, int] = {}
        for sp in ctx.subpaths:
            s = sp.features.get("shape_class", "organic")
            shape_dist[s] = shape_dist.get(s, 0) + 1

        fill_pct = 0.0
        if ctx.subpaths:
            fill_pct = ctx.subpaths[0].features.get("positive_fill_pct", 0.0)

        learnings = store.get_relevant_learnings(
            element_count=n_elements,
            symmetry_score=ctx.symmetry_score,
            fill_pct=fill_pct,
            composition_type=interp.composition_type,
            shape_distribution=shape_dist,
        )
        if learnings:
            lines.append("LEARNED PATTERNS:")
            for i, learning in enumerate(learnings, 1):
                lines.append(f"  {i}. {learning}")
            lines.append("")
    except Exception:
        pass  # Learning system is optional

    lines.append("=== END ENRICHMENT ===")
    return "\n".join(lines)

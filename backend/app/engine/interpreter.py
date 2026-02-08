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

    silhouette: str = ""
    pose: str = ""
    composition_type: str = ""
    mass_distribution: str = ""
    focal_elements: list[str] = field(default_factory=list)
    appendages: list[str] = field(default_factory=list)
    reading_hints: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = ["SPATIAL INTERPRETATION:"]
        if self.composition_type:
            lines.append(f"  Composition: {self.composition_type}")
        if self.pose:
            lines.append(f"  Pose: {self.pose}")
        if self.silhouette:
            lines.append(f"  Silhouette: {self.silhouette}")
        if self.mass_distribution:
            lines.append(f"  Mass distribution: {self.mass_distribution}")
        if self.appendages:
            lines.append("  Appendages/protrusions:")
            for a in self.appendages:
                lines.append(f"    - {a}")
        if self.focal_elements:
            lines.append("  Focal elements:")
            for f in self.focal_elements:
                lines.append(f"    - {f}")
        if self.reading_hints:
            lines.append("  Reading hints:")
            for h in self.reading_hints:
                lines.append(f"    - {h}")
        return "\n".join(lines)


def interpret(ctx: PipelineContext) -> SpatialInterpretation:
    """Synthesize high-level spatial interpretation from pipeline context."""
    grid = _parse_grid(ctx.ascii_grid_positive)

    return SpatialInterpretation(
        silhouette=_interpret_silhouette(ctx, grid),
        pose=_interpret_pose(ctx, grid),
        composition_type=_interpret_composition(ctx),
        mass_distribution=_analyze_mass_distribution(grid),
        focal_elements=_identify_focal_elements(ctx),
        appendages=_detect_appendages(ctx, grid),
        reading_hints=_generate_reading_hints(ctx, grid),
    )


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

    # Widest/narrowest points
    if col_profile:
        widest_row = int(np.argmax(row_profile))
        narrowest_filled = [i for i, v in enumerate(row_profile) if v > 10]
        if narrowest_filled:
            narrowest_row = narrowest_filled[np.argmin([row_profile[i] for i in narrowest_filled])]
            row_count = len(row_profile)
            widest_pos = "top" if widest_row < row_count * 0.33 else ("middle" if widest_row < row_count * 0.66 else "bottom")
            parts.append(f"widest at {widest_pos}")

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

    return "; ".join(parts)


def _interpret_pose(ctx: PipelineContext, grid: list[list[bool]]) -> str:
    """Determine pose/orientation from symmetry score + mass distribution."""
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
        parts.append("front-facing / bilateral")
        parts.append(f"strong {sym_axis} symmetry ({sym_score:.2f})")
    elif sym_score >= 0.5:
        parts.append("semi-symmetric")
        if abs(balance - 0.5) > 0.1:
            facing = "left-facing" if balance > 0.55 else "right-facing"
            parts.append(f"likely {facing} or 3/4 view")
        else:
            parts.append("roughly centered but with asymmetric features")
        parts.append(f"moderate {sym_axis} symmetry ({sym_score:.2f})")
    elif sym_score >= 0.25:
        if abs(balance - 0.5) > 0.15:
            facing = "left-facing" if balance > 0.55 else "right-facing"
            parts.append(f"side-profile, {facing}")
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

    return "; ".join(parts)


def _identify_focal_elements(ctx: PipelineContext) -> list[str]:
    """Find elements that likely represent key visual features."""
    focal = []

    for sp in ctx.subpaths:
        circ = sp.features.get("circularity", 0.0)
        tier = sp.features.get("size_tier", "MEDIUM")
        shape = sp.features.get("shape_class", "organic")
        cx, cy = sp.centroid

        # High circularity small elements = potential eyes, buttons, dots
        if circ > 0.75 and tier in ("SMALL", "MEDIUM"):
            contained_by = sp.features.get("contained_by", [])
            if contained_by:
                focal.append(
                    f"{sp.id}: circular ({shape}, circ={circ:.2f}) inside {contained_by[0]} — possible eye, button, or detail"
                )
            else:
                focal.append(
                    f"{sp.id}: circular ({shape}, circ={circ:.2f}) at ({cx:.0f},{cy:.0f}) — standalone focal point"
                )

        # Isolated small elements near body boundaries = held objects
        is_isolated = sp.features.get("is_isolated", False)
        if is_isolated and tier == "SMALL":
            focal.append(
                f"{sp.id}: isolated small element at ({cx:.0f},{cy:.0f}) — possible held object or detail"
            )

    # Limit to most interesting
    return focal[:8]


def _detect_appendages(ctx: PipelineContext, grid: list[list[bool]]) -> list[str]:
    """Detect protrusions/appendages from the main body."""
    appendages = []

    if not ctx.subpaths:
        return appendages

    # Find the largest element (main body)
    sorted_by_area = sorted(ctx.subpaths, key=lambda sp: sp.area, reverse=True)
    main_body = sorted_by_area[0]
    main_bbox = main_body.bbox  # (x1, y1, x2, y2)
    canvas_w, canvas_h = ctx.canvas_width, ctx.canvas_height

    for sp in ctx.subpaths:
        if sp.id == main_body.id:
            continue

        cx, cy = sp.centroid
        tier = sp.features.get("size_tier", "MEDIUM")
        shape = sp.features.get("shape_class", "organic")
        turning = sp.features.get("turning_classification", "")
        contained_by = sp.features.get("contained_by", [])

        # Skip elements contained deep within the body
        if main_body.id in contained_by:
            continue

        # Elements above the main body = ears, crown, tail plume
        if cy < main_bbox[1] and sp.area > main_body.area * 0.05:
            if sp.features.get("convexity", 1.0) < 0.6:
                appendages.append(
                    f"{sp.id}: large non-convex shape above main body — possible plume, fan, or flowing feature"
                )
            elif sp.features.get("aspect_ratio", 1.0) > 1.5:
                appendages.append(
                    f"{sp.id}: wide shape above main body — possible crown, hat, or spread feature"
                )
            else:
                appendages.append(
                    f"{sp.id}: shape above main body at ({cx:.0f},{cy:.0f}) — possible ear, horn, or top feature"
                )

        # Elements beside the main body = arms, wings, side features
        if cx < main_bbox[0] - canvas_w * 0.05 and tier in ("MEDIUM", "LARGE"):
            appendages.append(
                f"{sp.id}: extends left of main body — possible limb, wing, or side feature"
            )
        elif cx > main_bbox[2] + canvas_w * 0.05 and tier in ("MEDIUM", "LARGE"):
            appendages.append(
                f"{sp.id}: extends right of main body — possible limb, wing, or side feature"
            )

        # Elements below the main body = feet, base, tail
        if cy > main_bbox[3] and tier in ("MEDIUM", "LARGE"):
            appendages.append(
                f"{sp.id}: below main body — possible feet, base, or tail"
            )

    # Check for concentric groups that suggest eyes
    concentric = ctx.subpaths[0].features.get("concentric_groups", []) if ctx.subpaths else []
    for group in concentric:
        members = group.get("members", [])
        center = group.get("center", (0, 0))
        if len(members) >= 2:
            # Check if the members are small circular elements
            member_sps = [sp for sp in ctx.subpaths if sp.id in members]
            avg_circ = np.mean([sp.features.get("circularity", 0) for sp in member_sps]) if member_sps else 0
            if avg_circ > 0.5:
                appendages.append(
                    f"Concentric group [{', '.join(members)}] at ({center[0]:.0f},{center[1]:.0f}) — possible eye, wheel, or target"
                )

    return appendages[:10]


def _generate_reading_hints(ctx: PipelineContext, grid: list[list[bool]]) -> list[str]:
    """Generate plain-language hints for interpreting the SVG."""
    hints = []
    n = len(ctx.subpaths)

    # Symmetry hint
    sym_score = ctx.symmetry_score
    if sym_score < 0.5:
        hints.append(
            f"Low symmetry ({sym_score:.2f}) — this is likely a side-view, angled pose, "
            "or asymmetric composition, NOT a front-facing symmetric figure."
        )
    elif sym_score > 0.8:
        hints.append(
            f"High symmetry ({sym_score:.2f}) — this is likely a front-facing view, "
            "symmetric icon, or balanced design."
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

    # Element count hint
    if n > 40:
        hints.append(
            f"With {n} elements, this is a detailed illustration — "
            "look for layered body parts, shading regions, and decorative details."
        )
    elif n > 15:
        hints.append(
            f"With {n} elements, this is a moderately detailed design — "
            "each element likely represents a distinct visual feature."
        )

    # Containment depth hint
    if ctx.containment_matrix is not None:
        max_depth = 0
        for i in range(n):
            depth = sum(1 for j in range(n) if j != i and ctx.containment_matrix[j][i])
            max_depth = max(max_depth, depth)
        if max_depth >= 3:
            hints.append(
                f"Deep nesting ({max_depth} levels) suggests detailed anatomy — "
                "e.g., body → head → eye → pupil hierarchies."
            )

    # Cluster hint
    if ctx.cluster_labels is not None:
        unique = set(int(l) for l in ctx.cluster_labels if l >= 0)
        if len(unique) > 2:
            hints.append(
                f"{len(unique)} spatial clusters — elements form {len(unique)} "
                "distinct visual groups. Consider each cluster as a body part or feature group."
            )

    # Shape variety hint
    shapes = {}
    for sp in ctx.subpaths:
        s = sp.features.get("shape_class", "organic")
        shapes[s] = shapes.get(s, 0) + 1
    if "circular" in shapes and shapes["circular"] >= 2:
        hints.append(
            f"{shapes['circular']} circular elements — could be eyes, wheels, "
            "buttons, dots, or other round features."
        )

    # Overlap density hint
    high_overlap_count = 0
    for sp in ctx.subpaths:
        overlaps = sp.features.get("overlaps", [])
        high_overlaps = [o for o in overlaps if o.get("iou", 0) > 0.3]
        high_overlap_count += len(high_overlaps)
    if high_overlap_count > 10:
        hints.append(
            "Many overlapping elements — this design uses layered shapes "
            "(like shading, gradients, or body part overlaps) rather than discrete non-overlapping pieces."
        )

    # Grid shape hint
    if grid:
        col_profile = _column_profile(grid)
        row_profile = _row_profile(grid)
        # Check for a clear silhouette shape
        filled_cols = [i for i, v in enumerate(col_profile) if v > 15]
        if filled_cols:
            left_edge = filled_cols[0]
            right_edge = filled_cols[-1]
            span = right_edge - left_edge + 1
            total_cols = len(col_profile)
            if span < total_cols * 0.7:
                side = "left" if left_edge < total_cols * 0.2 else ("right" if right_edge > total_cols * 0.8 else "center")
                hints.append(
                    f"The silhouette occupies {span}/{total_cols} columns, "
                    f"positioned toward the {side} — not a full-width composition."
                )

    return hints

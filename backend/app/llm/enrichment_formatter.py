"""PipelineContext → enrichment text (~1,200 tokens) and EnrichmentOutput model."""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.models.enrichment import (
    ClusterInfo,
    ComponentInfo,
    ContainmentRelation,
    ElementSummary,
    EnrichmentOutput,
    SilhouetteInfo,
    SymmetryInfo,
)


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


def context_to_enrichment_text(ctx: PipelineContext) -> str:
    """Format PipelineContext as enrichment text block for LLM injection.

    Surfaces data from all 5 layers (61 transforms) in a compact format.
    Follows the format from docs/vectorsight_guide.md §3.2.
    """
    lines = ["=== VECTORSIGHT ENRICHMENT (auto-computed, 61 transforms) ===", ""]
    lines.append(f"ELEMENTS: {len(ctx.subpaths)} paths")
    lines.append(f"CANVAS: {ctx.canvas_width}×{ctx.canvas_height}")
    lines.append(f"TYPE: {'stroke-based' if ctx.is_stroke_based else 'fill-based'}")
    lines.append("")

    # ── Per-element summary (L1 transforms) ──
    sorted_sps = sorted(ctx.subpaths, key=lambda sp: sp.area, reverse=True)
    top_n = min(15, len(sorted_sps))

    lines.append(f"PER-ELEMENT (top {top_n} by area):")
    for sp in sorted_sps[:top_n]:
        shape_class = sp.features.get("shape_class", "organic")
        turning = sp.features.get("turning_classification", "")
        circ = sp.features.get("circularity", 0.0)
        conv = sp.features.get("convexity", 0.0)
        aspect = sp.features.get("aspect_ratio", 1.0)
        tier = sp.features.get("size_tier", "MEDIUM")
        corner_count = sp.features.get("corner_count", 0)
        has_gaps = sp.features.get("has_internal_gaps", False)
        cx, cy = sp.centroid
        x1, y1, x2, y2 = sp.bbox

        # Build compact per-element line
        parts = [
            f"  {sp.id}: [{tier}] {shape_class}",
        ]
        if turning:
            parts[0] += f"({turning})"
        parts.append(f"area={sp.area:.1f}")
        parts.append(f"bbox({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        parts.append(f"centroid({cx:.1f},{cy:.1f})")
        parts.append(f"circ={circ:.2f}, conv={conv:.2f}, aspect={aspect:.2f}")
        if corner_count > 0:
            parts.append(f"corners={corner_count}")
        if has_gaps:
            parts.append("GAPS")
        lines.append(", ".join(parts))
    lines.append("")

    # ── Nearest neighbors (T3.02) ──
    has_neighbors = any(sp.features.get("nearest_neighbors") for sp in ctx.subpaths)
    if has_neighbors:
        lines.append("NEAREST NEIGHBORS:")
        for sp in sorted_sps[:top_n]:
            neighbors = sp.features.get("nearest_neighbors", [])[:3]
            if neighbors:
                nn_strs = [f"{nb['element']}({nb['distance']:.1f}px)" for nb in neighbors]
                lines.append(f"  {sp.id} → {', '.join(nn_strs)}")
        lines.append("")

    # ── Relative positions (T3.05) ──
    has_positions = any(sp.features.get("relative_positions") for sp in ctx.subpaths)
    if has_positions and len(ctx.subpaths) >= 2:
        lines.append("RELATIVE POSITIONS:")
        for sp in sorted_sps[:top_n]:
            positions = sp.features.get("relative_positions", [])
            if positions:
                pos_strs = [f"{p['element']} is {p['direction']}" for p in positions[:4]]
                lines.append(f"  from {sp.id}: {', '.join(pos_strs)}")
        lines.append("")

    # ── Alignments (T3.03) ──
    h_groups: dict[str, list[str]] = {}
    v_groups: dict[str, list[str]] = {}
    for sp in ctx.subpaths:
        for alignment in sp.features.get("alignments", []):
            atype = alignment.get("type", "")
            others = alignment.get("with", [])
            key = f"{sp.id},{','.join(others)}"
            if atype == "horizontal":
                group_key = f"y≈{sp.centroid[1]:.0f}"
                if group_key not in h_groups:
                    h_groups[group_key] = [sp.id] + others
            elif atype == "vertical":
                group_key = f"x≈{sp.centroid[0]:.0f}"
                if group_key not in v_groups:
                    v_groups[group_key] = [sp.id] + others
    if h_groups or v_groups:
        lines.append("ALIGNMENTS:")
        for coord, members in v_groups.items():
            unique = list(dict.fromkeys(members))
            lines.append(f"  Vertical {coord}: [{', '.join(unique)}]")
        for coord, members in h_groups.items():
            unique = list(dict.fromkeys(members))
            lines.append(f"  Horizontal {coord}: [{', '.join(unique)}]")
        lines.append("")

    # ── Overlaps (T3.06) ──
    overlap_pairs: set[tuple[str, str]] = set()
    for sp in ctx.subpaths:
        for ov in sp.features.get("overlaps", []):
            pair = tuple(sorted([sp.id, ov["element"]]))
            if ov.get("iou", 0) > 0.01:
                overlap_pairs.add((pair[0], pair[1], str(ov["iou"])))
    if overlap_pairs:
        lines.append("OVERLAPS:")
        for a, b, iou in sorted(overlap_pairs):
            lines.append(f"  {a}↔{b} (IoU={iou})")
        lines.append("")

    # ── Containment (T3.01) ──
    if ctx.containment_matrix is not None:
        lines.append("CONTAINMENT:")
        n = len(ctx.subpaths)
        for i in range(n):
            children = [
                ctx.subpaths[j].id
                for j in range(n)
                if i != j and ctx.containment_matrix[i][j]
            ]
            if children:
                lines.append(f"  {ctx.subpaths[i].id} contains: {', '.join(children)}")
        lines.append("")

    # ── Symmetry (T1.20) ──
    if ctx.symmetry_axis:
        lines.append(f"SYMMETRY: {ctx.symmetry_axis} (score {ctx.symmetry_score:.2f})")
        if ctx.symmetry_pairs:
            pair_strs = [
                f"({ctx.subpaths[a].id}↔{ctx.subpaths[b].id})"
                for a, b in ctx.symmetry_pairs
                if a < len(ctx.subpaths) and b < len(ctx.subpaths)
            ]
            lines.append(f"  Pairs: {', '.join(pair_strs)}")
        lines.append("")

    # ── Concentric groups (T3.15) ──
    concentric = []
    if ctx.subpaths:
        concentric = ctx.subpaths[0].features.get("concentric_groups", [])
    if concentric:
        lines.append("CONCENTRIC:")
        for group in concentric:
            members = group.get("members", [])
            center = group.get("center", (0, 0))
            lines.append(f"  [{', '.join(members)}] share center at ({center[0]},{center[1]})")
        lines.append("")

    # ── Repeated elements (T3.09 + T3.14) ──
    rep_groups = []
    if ctx.subpaths:
        rep_groups = ctx.subpaths[0].features.get("repetition_groups", [])
    if rep_groups:
        lines.append("REPEATED ELEMENTS:")
        for group in rep_groups:
            members = group.get("members", [])
            count = group.get("count", len(members))
            shape = group.get("shape_class", "unknown")
            pattern = group.get("pattern", "")
            line = f"  [{', '.join(members)}] ×{count} {shape}"
            if pattern and pattern != "none":
                line += f", pattern={pattern}"
            # Angular spacing from T3.16
            spacing = group.get("angular_spacing")
            if spacing:
                mean = spacing.get("mean_spacing", 0)
                regular = spacing.get("is_regular", False)
                line += f", spacing={mean:.0f}°"
                if regular:
                    line += " (regular)"
            lines.append(line)
        lines.append("")

    # ── Clusters (T3.07) ──
    if ctx.cluster_labels is not None:
        lines.append("SPATIAL CLUSTERS:")
        labels = ctx.cluster_labels
        unique_labels = set(int(l) for l in labels if l >= 0)
        for label in sorted(unique_labels):
            members = [ctx.subpaths[i].id for i in range(len(labels)) if labels[i] == label]
            if members:
                lines.append(f"  Cluster {label}: [{', '.join(members)}]")
        noise = [ctx.subpaths[i].id for i in range(len(labels)) if labels[i] < 0]
        if noise:
            lines.append(f"  Isolated: {', '.join(noise)}")
        lines.append("")

    # ── Size tiers ──
    tiers: dict[str, list[str]] = {"LARGE": [], "MEDIUM": [], "SMALL": []}
    for sp in ctx.subpaths:
        tier = sp.features.get("size_tier", "MEDIUM")
        tiers.setdefault(tier, []).append(sp.id)
    has_tiers = any(v for v in tiers.values())
    if has_tiers:
        lines.append("SIZE TIERS:")
        for tier_name in ["LARGE", "MEDIUM", "SMALL"]:
            if tiers.get(tier_name):
                lines.append(f"  {tier_name}: {', '.join(tiers[tier_name])}")
        lines.append("")

    # ── Directional coverage (T1.06) — compact per-element ──
    has_directional = any(
        sp.features.get("directional_coverage_pct") is not None
        for sp in ctx.subpaths
    )
    if has_directional:
        partial = [
            sp for sp in sorted_sps[:top_n]
            if sp.features.get("directional_coverage_pct", 100) < 100
        ]
        if partial:
            lines.append("DIRECTIONAL COVERAGE:")
            for sp in partial:
                pct = sp.features.get("directional_coverage_pct", 100)
                gap = sp.features.get("directional_gap", 0)
                bins = sp.features.get("directional_bins", {})
                empty = [d for d, v in bins.items() if v == 0]
                line = f"  {sp.id}: {pct:.0f}% coverage"
                if gap > 0:
                    line += f", {gap:.0f}° gap"
                if empty:
                    line += f", empty: {','.join(empty)}"
                lines.append(line)
            lines.append("")

    # ── Topology & structure (T3.10) ──
    if ctx.subpaths:
        topo = ctx.subpaths[0].features.get("topology_type")
        if topo:
            lines.append(f"TOPOLOGY: {topo}")
            patterns = ctx.subpaths[0].features.get("topology_patterns", [])
            for p in patterns[:5]:
                lines.append(f"  {p}")
            lines.append("")

    # ── Stacking tree (T3.21) ──
    if ctx.subpaths:
        tree_text = ctx.subpaths[0].features.get("stacking_tree_text", "")
        if tree_text:
            lines.append("STACKING TREE:")
            lines.append(tree_text)
            lines.append("")

    # ── Structural pattern report (T3.19) ──
    if ctx.subpaths:
        report = ctx.subpaths[0].features.get("structural_pattern_report", "")
        if report:
            lines.append(f"STRUCTURE: {report}")
            lines.append("")

    # ── Connected components (T3.18) ──
    if ctx.component_labels:
        n_comp = max(ctx.component_labels) + 1
        if n_comp > 1:
            lines.append(f"VISUAL UNITS: {n_comp} connected components")
            lines.append("")

    # ── Figure-ground (T2.07) ──
    if ctx.subpaths:
        fg_type = ctx.subpaths[0].features.get("figure_ground_type")
        fg_desc = ctx.subpaths[0].features.get("figure_ground_description", "")
        fill_pct = ctx.subpaths[0].features.get("positive_fill_pct")
        if fg_type:
            line = f"FIGURE-GROUND: {fg_type}"
            if fill_pct is not None:
                line += f" ({fill_pct:.0f}% fill)"
            if fg_desc:
                line += f" — {fg_desc}"
            lines.append(line)
            lines.append("")

    # ── ASCII grids (T2.01) ──
    if ctx.ascii_grid_positive:
        lines.append("POSITIVE SPACE GRID:")
        lines.append(ctx.ascii_grid_positive)
        lines.append("")

    if ctx.ascii_grid_negative:
        lines.append("NEGATIVE SPACE GRID:")
        lines.append(ctx.ascii_grid_negative)
        lines.append("")

    # ── Validation summary (T4.05) ──
    if ctx.subpaths:
        issues = []
        for sp in ctx.subpaths:
            sp_issues = sp.features.get("consistency_issues", [])
            for issue in sp_issues:
                issues.append(f"  {sp.id}: {issue}")
        if issues:
            lines.append("VALIDATION ISSUES:")
            for issue in issues[:5]:
                lines.append(issue)
            lines.append("")

    # ── Spatial Interpretation (synthesized from all transforms) ──
    from app.engine.interpreter import interpret

    interp = interpret(ctx)
    interp_text = interp.to_text()
    if interp_text:
        lines.append(interp_text)
        lines.append("")

    # ── Learned patterns (from past sessions) ──
    try:
        from app.learning.memory import get_memory_store

        store = get_memory_store()
        # Build shape distribution for matching
        shape_dist: dict[str, int] = {}
        for sp in ctx.subpaths:
            s = sp.features.get("shape_class", "organic")
            shape_dist[s] = shape_dist.get(s, 0) + 1

        fill_pct = 0.0
        if ctx.subpaths:
            fill_pct = ctx.subpaths[0].features.get("positive_fill_pct", 0.0)

        learnings = store.get_relevant_learnings(
            element_count=len(ctx.subpaths),
            symmetry_score=ctx.symmetry_score,
            fill_pct=fill_pct,
            composition_type=interp.composition_type,
            shape_distribution=shape_dist,
        )
        if learnings:
            lines.append("LEARNED PATTERNS (from past sessions):")
            for i, learning in enumerate(learnings, 1):
                lines.append(f"  {i}. {learning}")
            lines.append("")
    except Exception:
        pass  # Learning system is optional

    lines.append("=== END ENRICHMENT ===")
    return "\n".join(lines)

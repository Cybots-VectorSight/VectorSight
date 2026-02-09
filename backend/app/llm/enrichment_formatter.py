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
    Order: interpretation first (highest signal), then grids, then details.
    """
    from app.engine.interpreter import interpret
    from app.utils.rasterizer import (
        composite_braille_grid,
        element_braille_grid,
        element_mini_grid,
        group_braille_grid,
    )

    lines = ["=== VECTORSIGHT ENRICHMENT ===", ""]
    n_elements = len(ctx.subpaths)
    lines.append(f"ELEMENTS: {n_elements} paths | CANVAS: {ctx.canvas_width:.0f}\u00d7{ctx.canvas_height:.0f} | TYPE: {'stroke-based' if ctx.is_stroke_based else 'fill-based'}")
    lines.append("")

    sorted_sps = sorted(ctx.subpaths, key=lambda sp: sp.area, reverse=True)
    top_n = min(15, len(sorted_sps))

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: SPATIAL INTERPRETATION (highest-value signal, first)
    # ═══════════════════════════════════════════════════════════════════

    interp = interpret(ctx)
    interp_text = interp.to_text()
    if interp_text:
        lines.append(interp_text)
        lines.append("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: VISUAL OVERVIEW (grids + structure)
    # ═══════════════════════════════════════════════════════════════════

    # ── Braille composite grid (8x denser than text, ~32×16 chars for 64×64) ──
    braille_composite = composite_braille_grid(ctx.subpaths, ctx.canvas_width, ctx.canvas_height, resolution=64)
    if braille_composite:
        lines.append("SILHOUETTE (Braille, each char=2×4 pixels, dots=filled):")
        lines.append(braille_composite)
        lines.append("")
    elif ctx.ascii_grid_halfblock:
        lines.append("POSITIVE SPACE GRID (\u2588=filled):")
        lines.append(ctx.ascii_grid_halfblock)
        lines.append("")

    # ── Stacking tree (T3.21) — structural hierarchy ──
    if ctx.subpaths:
        tree_text = ctx.subpaths[0].features.get("stacking_tree_text", "")
        if tree_text:
            lines.append("STACKING TREE:")
            lines.append(tree_text)
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
            pair_strs = []
            for a, b in ctx.symmetry_pairs:
                if a >= len(ctx.subpaths) or b >= len(ctx.subpaths):
                    continue
                sp_a, sp_b = ctx.subpaths[a], ctx.subpaths[b]
                circ_a = sp_a.features.get("circularity", 0)
                circ_b = sp_b.features.get("circularity", 0)
                shape_a = sp_a.features.get("shape_class", "organic")
                shape_b = sp_b.features.get("shape_class", "organic")
                if shape_a != shape_b or abs(circ_a - circ_b) > 0.3:
                    pair_strs.append(f"({sp_a.id}\u2194{sp_b.id} DIFF:{shape_a}/{shape_b})")
                else:
                    pair_strs.append(f"({sp_a.id}\u2194{sp_b.id})")
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
            lines.append(f"  [{', '.join(members)}] share center at ({center[0]:.0f},{center[1]:.0f})")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: ELEMENT DETAILS (integer coordinates)
    # ═══════════════════════════════════════════════════════════════════

    # ── Per-element summary (L1 transforms) — integer coords ──
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

        parts = [
            f"  {sp.id}: [{tier}] {shape_class}",
        ]
        if turning:
            parts[0] += f"({turning})"
        parts.append(f"area={sp.area:.0f}")
        parts.append(f"bbox({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
        parts.append(f"centroid({int(cx)},{int(cy)})")
        parts.append(f"circ={circ:.2f}, conv={conv:.2f}, aspect={aspect:.2f}")
        if corner_count > 0:
            parts.append(f"corners={corner_count}")
        if has_gaps:
            parts.append("GAPS")
        lines.append(", ".join(parts))
    lines.append("")

    # ── Element shape Braille grids (top 3 largest) ──
    mini_grid_sps = sorted_sps[:3]
    has_mini = False
    mini_lines = ["ELEMENT SHAPES (top 3, Braille 2×4 dots/char):"]
    for sp in mini_grid_sps:
        if len(sp.points) < 5:
            continue
        mini = element_braille_grid(sp.points, sp.bbox, resolution=48)
        if mini:
            has_mini = True
            tier = sp.features.get("size_tier", "MEDIUM")
            shape_class = sp.features.get("shape_class", "organic")
            mini_lines.append(f"  {sp.id} ({tier}, {shape_class}):")
            for mline in mini.split("\n"):
                mini_lines.append(f"    {mline}")
            mini_lines.append("")
    if has_mini:
        lines.extend(mini_lines)
        lines.append("")

    # ── Concentric group Braille grids ──
    # Render each concentric group as a combined Braille mini-grid
    # This reveals WHAT each group is (eye, acorn, wheel, button, etc.)
    # Prioritize: more members first, skip canvas-edge groups, limit to 4
    if concentric:
        # Filter and sort: skip canvas edge, sort by member count desc
        cw, ch = ctx.canvas_width, ctx.canvas_height
        scored_groups = []
        for group in concentric:
            members = group.get("members", [])
            center = group.get("center", (0, 0))
            if len(members) < 2:
                continue
            # Skip groups at canvas edge (center within 2% of edge)
            cx_pct = center[0] / cw * 100 if cw > 0 else 50
            cy_pct = center[1] / ch * 100 if ch > 0 else 50
            if cx_pct < 2 or cx_pct > 98 or cy_pct < 2 or cy_pct > 98:
                continue
            scored_groups.append((len(members), group))
        scored_groups.sort(key=lambda x: x[0], reverse=True)

        group_lines = ["CONCENTRIC GROUP SHAPES (Braille, combined members):"]
        has_group_grids = False
        for _, group in scored_groups[:4]:
            members = group.get("members", [])
            center = group.get("center", (0, 0))
            member_sps = [sp for sp in ctx.subpaths if sp.id in members]
            if not member_sps:
                continue
            braille = group_braille_grid(member_sps, resolution=48)
            if braille:
                has_group_grids = True
                member_str = "+".join(members)
                group_lines.append(f"  [{member_str}] at ({center[0]:.0f},{center[1]:.0f}):")
                for bline in braille.split("\n"):
                    group_lines.append(f"    {bline}")
                group_lines.append("")
        if has_group_grids:
            lines.extend(group_lines)
            lines.append("")

    # ── Cluster scene summary ──
    if interp.cluster_scene:
        lines.append("CLUSTER SCENE:")
        for desc in interp.cluster_scene:
            lines.append(f"  {desc}")
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
                line += f" \u2014 {fg_desc}"
            lines.append(line)
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

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: RELATIONSHIPS
    # ═══════════════════════════════════════════════════════════════════

    # ── Nearest neighbors (T3.02) ──
    has_neighbors = any(sp.features.get("nearest_neighbors") for sp in ctx.subpaths)
    if has_neighbors:
        zero_count = 0
        total_count = 0
        for sp in sorted_sps[:top_n]:
            for nb in sp.features.get("nearest_neighbors", [])[:3]:
                total_count += 1
                if nb["distance"] < 0.1:
                    zero_count += 1
        if total_count > 0 and zero_count / total_count > 0.8:
            lines.append("NEAREST NEIGHBORS: most elements touching/overlapping (layered illustration)")
        else:
            lines.append("NEAREST NEIGHBORS:")
            for sp in sorted_sps[:top_n]:
                neighbors = sp.features.get("nearest_neighbors", [])[:3]
                if neighbors:
                    nn_strs = [f"{nb['element']}({nb['distance']:.1f}px)" for nb in neighbors]
                    lines.append(f"  {sp.id} \u2192 {', '.join(nn_strs)}")
        lines.append("")

    # ── Relative positions (T3.05) ──
    has_positions = any(sp.features.get("relative_positions") for sp in ctx.subpaths)
    if has_positions and len(ctx.subpaths) >= 2:
        tier_lookup = {sp.id: sp.features.get("size_tier", "MEDIUM") for sp in ctx.subpaths}
        pos_show = min(5, top_n) if n_elements > 30 else top_n
        lines.append(f"RELATIVE POSITIONS (top {pos_show} by area):")
        for sp in sorted_sps[:pos_show]:
            positions = sp.features.get("relative_positions", [])
            filtered = [p for p in positions if tier_lookup.get(p["element"], "MEDIUM") != "SMALL"]
            if filtered:
                pos_strs = [f"{p['element']} is {p['direction']}" for p in filtered[:4]]
                lines.append(f"  from {sp.id}: {', '.join(pos_strs)}")
        lines.append("")

    # ── Overlaps (T3.06) — capped ──
    all_overlaps: list[tuple[str, str, float]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for sp in ctx.subpaths:
        for ov in sp.features.get("overlaps", []):
            pair = tuple(sorted([sp.id, ov["element"]]))
            iou = ov.get("iou", 0)
            if iou > 0.01 and pair not in seen_pairs:
                all_overlaps.append((pair[0], pair[1], iou))
                seen_pairs.add(pair)
    if all_overlaps:
        all_overlaps.sort(key=lambda x: x[2], reverse=True)
        max_show = min(30, len(all_overlaps))
        shown = all_overlaps[:max_show]
        if max_show < len(all_overlaps):
            lines.append(f"OVERLAPS (top {max_show} by IoU, {len(all_overlaps)} total):")
        else:
            lines.append("OVERLAPS:")
        for a, b, iou in shown:
            lines.append(f"  {a}\u2194{b} (IoU={iou:.3f})")
        lines.append("")

    # ── Alignments (T3.03) ──
    h_groups: dict[str, list[str]] = {}
    v_groups: dict[str, list[str]] = {}
    for sp in ctx.subpaths:
        for alignment in sp.features.get("alignments", []):
            atype = alignment.get("type", "")
            others = alignment.get("with", [])
            if atype == "horizontal":
                group_key = f"y\u2248{sp.centroid[1]:.0f}"
                if group_key not in h_groups:
                    h_groups[group_key] = [sp.id] + others
            elif atype == "vertical":
                group_key = f"x\u2248{sp.centroid[0]:.0f}"
                if group_key not in v_groups:
                    v_groups[group_key] = [sp.id] + others

    min_align_group = 3 if n_elements > 15 else 2

    def _dedup_alignment_groups(
        groups: dict[str, list[str]], min_size: int
    ) -> dict[str, list[str]]:
        """Merge groups sharing >70% of members, keep the largest."""
        filtered = {k: list(dict.fromkeys(v)) for k, v in groups.items()
                    if len(dict.fromkeys(v)) >= min_size}
        if not filtered:
            return {}
        items = sorted(filtered.items(), key=lambda x: len(x[1]), reverse=True)
        result: dict[str, list[str]] = {}
        used_members: list[set[str]] = []
        for coord, members in items:
            member_set = set(members)
            duplicate = False
            for kept in used_members:
                overlap = len(member_set & kept) / max(len(member_set), len(kept))
                if overlap > 0.7:
                    duplicate = True
                    break
            if not duplicate:
                result[coord] = members
                used_members.append(member_set)
        return result

    v_filtered = _dedup_alignment_groups(v_groups, min_align_group)
    h_filtered = _dedup_alignment_groups(h_groups, min_align_group)
    if v_filtered or h_filtered:
        lines.append(f"ALIGNMENTS (groups of {min_align_group}+):")
        for coord, members in v_filtered.items():
            lines.append(f"  Vertical {coord}: [{', '.join(members)}]")
        for coord, members in h_filtered.items():
            lines.append(f"  Horizontal {coord}: [{', '.join(members)}]")
        lines.append("")

    # ── Clusters (T3.07) ──
    if ctx.cluster_labels is not None:
        lines.append("SPATIAL CLUSTERS:")
        labels = ctx.cluster_labels
        unique_labels = set(int(l) for l in labels if l >= 0)
        for label in sorted(unique_labels):
            indices = [i for i in range(len(labels)) if labels[i] == label]
            members = [ctx.subpaths[i].id for i in indices]
            if members:
                cx = sum(ctx.subpaths[i].centroid[0] for i in indices) / len(indices)
                cy = sum(ctx.subpaths[i].centroid[1] for i in indices) / len(indices)
                lines.append(f"  Cluster {label} at ({cx:.0f},{cy:.0f}): [{', '.join(members)}]")
        noise = [ctx.subpaths[i].id for i in range(len(labels)) if labels[i] < 0]
        if noise:
            lines.append(f"  Isolated: {', '.join(noise)}")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: SUPPLEMENTARY
    # ═══════════════════════════════════════════════════════════════════

    # ── Repeated elements (T3.09 + T3.14) ──
    rep_groups = []
    if ctx.subpaths:
        rep_groups = ctx.subpaths[0].features.get("repetition_groups", [])
    if rep_groups:
        total_elements = len(ctx.subpaths)
        rep_groups = [g for g in rep_groups if g.get("count", len(g.get("members", []))) / total_elements <= 0.8]
    if rep_groups:
        lines.append("REPEATED ELEMENTS:")
        for group in rep_groups:
            members = group.get("members", [])
            count = group.get("count", len(members))
            shape = group.get("shape_class", "unknown")
            pattern = group.get("pattern", "")
            line = f"  [{', '.join(members)}] \u00d7{count} {shape}"
            if pattern and pattern != "none":
                line += f", pattern={pattern}"
            spacing = group.get("angular_spacing")
            if spacing:
                mean = spacing.get("mean_spacing", 0)
                regular = spacing.get("is_regular", False)
                line += f", spacing={mean:.0f}\u00b0"
                if regular:
                    line += " (regular)"
            lines.append(line)
        lines.append("")

    # ── Directional coverage (T1.06) ──
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
                    line += f", {gap:.0f}\u00b0 gap"
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

    # ── Connected components (T3.18) ──
    if ctx.component_labels:
        n_comp = max(ctx.component_labels) + 1
        if n_comp > 1:
            lines.append(f"VISUAL UNITS: {n_comp} connected components")
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

    # ── Learned patterns (from past sessions) ──
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

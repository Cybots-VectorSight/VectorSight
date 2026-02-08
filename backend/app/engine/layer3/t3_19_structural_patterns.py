"""T3.19 — Structural Pattern Report. ★★

Report recurring geometric arrangements factually: containment patterns,
radial arrangements, bilateral symmetry, appendages.
Geometry only, no semantic naming — let LLM interpret.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.19",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T3.01", "T3.07", "T3.08", "T3.09", "T3.14"],
    description="Generate factual structural pattern report",
)
def structural_patterns(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    patterns: list[str] = []

    # 1. Size distribution
    areas = [sp.area for sp in ctx.subpaths]
    max_area = max(areas) if areas else 0
    if max_area > 0:
        large = sum(1 for a in areas if a > max_area * 0.5)
        medium = sum(1 for a in areas if max_area * 0.1 < a <= max_area * 0.5)
        small = sum(1 for a in areas if a <= max_area * 0.1 and a > 0)
        parts = []
        if large:
            parts.append(f"{large} large element{'s' if large > 1 else ''}")
        if medium:
            parts.append(f"{medium} medium element{'s' if medium > 1 else ''}")
        if small:
            parts.append(f"{small} small element{'s' if small > 1 else ''}")
        if parts:
            patterns.append(" + ".join(parts))

    # 2. Containment structure
    cmat = ctx.containment_matrix
    if cmat is not None:
        containers = []
        for i in range(n):
            children = sum(1 for j in range(n) if cmat[i][j])
            if children > 0:
                containers.append((i, children))
        for idx, child_count in containers:
            sp = ctx.subpaths[idx]
            child_classes = []
            for j in range(n):
                if cmat[idx][j]:
                    child_classes.append(ctx.subpaths[j].features.get("shape_class", "?"))
            patterns.append(
                f"1 {sp.features.get('shape_class', '?')} containing "
                f"{child_count} inner element{'s' if child_count > 1 else ''} "
                f"({', '.join(child_classes)})"
            )

    # 3. Symmetry
    score = ctx.symmetry_score
    axis = ctx.symmetry_axis
    if score > 0.7 and axis:
        patterns.append(f"Bilateral symmetry score: {score:.2f} about {axis} axis")

    # 4. Repetition
    groups = ctx.subpaths[0].features.get("repetition_groups", []) if n > 0 else []
    for g in groups:
        if g["count"] >= 2:
            pattern_type = g.get("pattern", "")
            patterns.append(
                f"{g['count']}× repeated {g['shape_class']} elements"
                + (f" in {pattern_type} arrangement" if pattern_type else "")
            )

    # 5. Spatial distribution
    centroids = np.array([sp.centroid for sp in ctx.subpaths])
    center_x = ctx.canvas_width / 2
    center_y = ctx.canvas_height / 2

    upper = sum(1 for c in centroids if c[1] < center_y)
    lower = n - upper
    left = sum(1 for c in centroids if c[0] < center_x)
    right = n - left

    if upper > 0 and lower > 0:
        patterns.append(f"{upper} elements upper half + {lower} elements lower half")

    # Compile report
    report = "; ".join(patterns) if patterns else "No notable structural patterns"

    for sp in ctx.subpaths:
        sp.features["structural_pattern_report"] = report
        sp.features["structural_patterns"] = patterns

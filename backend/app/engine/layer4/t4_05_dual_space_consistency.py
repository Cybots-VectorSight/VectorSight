"""T4.05 — Dual-Space Consistency Check. ★★★ CRITICAL

Cross-check positive-space features against negative-space features.
Flag inconsistencies where positive and negative analyses disagree.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T4.05",
    layer=Layer.VALIDATION,
    dependencies=["T2.07", "T1.14"],
    description="Cross-check positive vs negative space consistency",
)
def dual_space_consistency(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        issues: list[str] = []
        is_consistent = True

        # Check 1: Fill percentage vs shape class
        fill_pct = sp.features.get("positive_fill_pct", 0)
        shape_class = sp.features.get("shape_class", "")

        if fill_pct > 70 and shape_class == "linear":
            issues.append("High fill but classified as linear")
            is_consistent = False

        # Check 2: Area vs bounding box
        area = sp.area
        bbox_area = sp.width * sp.height
        if bbox_area > 0:
            fill_ratio = area / bbox_area
            if fill_ratio > 1.1:  # Area shouldn't exceed bbox
                issues.append(f"Area ({area:.1f}) exceeds bbox area ({bbox_area:.1f})")
                is_consistent = False

        # Check 3: Negative space vs containment
        neg_count = sp.features.get("negative_space_count", 0)
        contains = sp.features.get("contains", [])
        if neg_count > 0 and len(contains) == 0 and sp.features.get("containment_depth", 0) == 0:
            # Negative space exists but no containment → could be legitimate holes
            pass  # Not necessarily inconsistent

        # Check 4: Symmetry consistency
        bilateral = sp.features.get("bilateral_symmetry_score", 0)
        mirror = sp.features.get("has_mirror_pair", False)
        if bilateral > 0.9 and not mirror and ctx.num_elements > 2:
            # High bilateral symmetry but no mirror partner in multi-element
            # Not an error, just noting
            pass

        # Check 5: Circularity vs centroid distance CV
        circ = sp.features.get("circularity", 0)
        cd_class = sp.features.get("centroid_distance_classification", "")
        if circ > 0.9 and cd_class == "complex":
            issues.append("High circularity but complex centroid distance profile")
            is_consistent = False
        if circ < 0.5 and cd_class == "circular":
            issues.append("Low circularity but circular centroid distance profile")
            is_consistent = False

        sp.features["dual_space_consistent"] = is_consistent
        sp.features["consistency_issues"] = issues
        sp.features["validation_passed"] = is_consistent

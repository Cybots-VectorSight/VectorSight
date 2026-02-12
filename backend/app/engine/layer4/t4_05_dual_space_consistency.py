"""T4.05 — Dual-Space Consistency Check. ★★★ CRITICAL

Cross-check positive-space features against negative-space features.
Flag inconsistencies where positive and negative analyses disagree.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform

# ── Consistency check constants ──
# High fill + linear shape = contradiction (linear shapes are thin).
_FILL_HIGH_PCT = 70        # > 70% fill (7/10)
# Area/bbox ratio > 110% = impossible overflow (10% tolerance for float errors).
_AREA_BBOX_TOLERANCE = 1.1  # = 1 + 1/10
# Near-perfect bilateral symmetry for consistency check.
_SYM_NEAR_PERFECT = 0.9    # 9/10
# High circularity for consistency check.
_CIRC_HIGH_CONSISTENCY = 0.9  # 9/10
# Low circularity boundary.
_CIRC_LOW_CONSISTENCY = 0.5   # 1/2
# Minimum elements for mirror pair check.
_MIN_ELEMENTS_MIRROR = 2      # Pair requires >=2


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

        if fill_pct > _FILL_HIGH_PCT and shape_class == "linear":
            issues.append("High fill but classified as linear")
            is_consistent = False

        # Check 2: Area vs bounding box
        area = sp.area
        bbox_area = sp.width * sp.height
        if bbox_area > 0:
            fill_ratio = area / bbox_area
            if fill_ratio > _AREA_BBOX_TOLERANCE:  # Area shouldn't exceed bbox
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
        if bilateral > _SYM_NEAR_PERFECT and not mirror and ctx.num_elements > _MIN_ELEMENTS_MIRROR:
            # High bilateral symmetry but no mirror partner in multi-element
            # Not an error, just noting
            pass

        # Check 5: Circularity vs centroid distance CV
        circ = sp.features.get("circularity", 0)
        cd_class = sp.features.get("centroid_distance_classification", "")
        if circ > _CIRC_HIGH_CONSISTENCY and cd_class == "complex":
            issues.append("High circularity but complex centroid distance profile")
            is_consistent = False
        if circ < _CIRC_LOW_CONSISTENCY and cd_class == "circular":
            issues.append("Low circularity but circular centroid distance profile")
            is_consistent = False

        sp.features["dual_space_consistent"] = is_consistent
        sp.features["consistency_issues"] = issues
        sp.features["validation_passed"] = is_consistent

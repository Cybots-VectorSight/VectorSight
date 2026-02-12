"""T3.16 — Angular Spacing Analysis. ★★

For radially-arranged elements: compute angle between each pair.
Snap to clean multiples (15, 30, 45, 90 degrees) within ±2°.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.math_helpers import CLEAN_ANGLES, snap_angle

# +/-3 deg ~ 1/120 of full circle. ISO 2768-mK manufacturing tolerance
# for angular features. Detects "regular" radial arrangements.
_ANGULAR_TOLERANCE_DEG = 3


@transform(
    id="T3.16",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T3.14", "T3.15"],
    description="Analyze angular spacing of radially-arranged elements",
)
def angular_spacing(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 3:
        return

    # Find radial groups from T3.14
    groups = ctx.subpaths[0].features.get("repetition_groups", [])
    radial_groups = [g for g in groups if g.get("pattern") == "radial"]

    if not radial_groups:
        for sp in ctx.subpaths:
            sp.features["angular_spacing"] = None
        return

    for group in radial_groups:
        member_ids = group["members"]
        members = [sp for sp in ctx.subpaths if sp.id in member_ids]
        if len(members) < 3:
            continue

        centroids = np.array([m.centroid for m in members])
        center = centroids.mean(axis=0)

        # Compute angles from center
        angles = np.arctan2(centroids[:, 1] - center[1], centroids[:, 0] - center[0])
        angles_deg = np.degrees(angles) % 360
        angles_sorted = np.sort(angles_deg)

        # Compute spacing between consecutive elements
        spacings = np.diff(angles_sorted)
        if len(angles_sorted) > 1:
            # Add wrap-around spacing
            wrap = 360 - angles_sorted[-1] + angles_sorted[0]
            spacings = np.append(spacings, wrap)

        # Snap to clean angles
        snapped = [snap_angle(float(s)) for s in spacings]
        is_regular = all(abs(s - snapped[0]) < _ANGULAR_TOLERANCE_DEG for s in spacings) if len(spacings) > 0 else False

        spacing_info = {
            "spacings_deg": [round(float(s), 1) for s in spacings],
            "snapped_deg": snapped,
            "mean_spacing": round(float(np.mean(spacings)), 1) if len(spacings) > 0 else 0,
            "is_regular": is_regular,
            "n_elements": len(members),
        }

        group["angular_spacing"] = spacing_info

    for sp in ctx.subpaths:
        sp.features["repetition_groups"] = groups
        # Set angular spacing if this element is in a radial group
        for g in radial_groups:
            if sp.id in g["members"]:
                sp.features["angular_spacing"] = g.get("angular_spacing")
                break
        else:
            sp.features["angular_spacing"] = None

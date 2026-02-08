"""T4.03 — Multi-Orientation Check. ★

Verify that the description works at 0°, 90°, 180°, 270°.
Compute how stable features are across rotations.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import centroid_distance_cv


@transform(
    id="T4.03",
    layer=Layer.VALIDATION,
    dependencies=["T1.04"],
    description="Check feature stability across 4 orientations",
)
def multi_orientation_check(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 5:
            sp.features["orientation_stability"] = 1.0
            continue

        pts = sp.points.copy()
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

        # Compute centroid-distance CV at 0, 90, 180, 270 degrees
        cvs = []
        for angle_deg in [0, 90, 180, 270]:
            rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rotated = pts.copy()
            rotated[:, 0] = (pts[:, 0] - cx) * cos_a - (pts[:, 1] - cy) * sin_a + cx
            rotated[:, 1] = (pts[:, 0] - cx) * sin_a + (pts[:, 1] - cy) * cos_a + cy
            cv = centroid_distance_cv(rotated)
            if cv != float("inf"):
                cvs.append(cv)

        if len(cvs) >= 2:
            # Stability = 1 - normalized variance of CVs
            mean_cv = np.mean(cvs)
            if mean_cv > 0:
                stability = 1.0 - float(np.std(cvs) / mean_cv)
            else:
                stability = 1.0
        else:
            stability = 1.0

        sp.features["orientation_stability"] = round(max(0.0, min(1.0, stability)), 3)

"""T1.21 — Corner / Vertex Detection (VTracer method). ★★

Compute signed rotation angle between consecutive edge vectors.
Track accumulated angle displacement. When |displacement| > threshold → corner.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.21",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Detect corners using VTracer signed-angle method",
)
def corner_detection(ctx: PipelineContext) -> None:
    threshold = 0.5  # radians (~28.6°)

    for sp in ctx.subpaths:
        if len(sp.points) < 4:
            sp.features["corner_count"] = 0
            sp.features["corner_angles"] = []
            continue

        pts = sp.points
        edges = np.diff(pts, axis=0)

        # Compute signed angles between consecutive edges
        corners: list[dict] = []
        displacement = 0.0

        for i in range(len(edges) - 1):
            e1 = edges[i]
            e2 = edges[i + 1]
            # Signed angle from e1 to e2
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            dot = e1[0] * e2[0] + e1[1] * e2[1]
            angle = np.arctan2(cross, dot)
            displacement += angle

            if abs(displacement) > threshold:
                corners.append({
                    "index": i + 1,
                    "angle": round(float(displacement) * 180 / np.pi, 1),
                    "x": round(float(pts[i + 1, 0]), 2),
                    "y": round(float(pts[i + 1, 1]), 2),
                })
                displacement = 0.0

        sp.features["corner_count"] = len(corners)
        sp.features["corner_angles"] = [c["angle"] for c in corners]
        sp.features["corner_positions"] = [(c["x"], c["y"]) for c in corners]

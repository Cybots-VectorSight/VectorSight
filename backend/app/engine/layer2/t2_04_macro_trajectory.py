"""T2.04 — Macro Trajectory Narrative. ★★

Natural language description of composite silhouette boundary direction.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


DIRECTION_NAMES = {
    0: "right", 1: "upper-right", 2: "up", 3: "upper-left",
    4: "left", 5: "lower-left", 6: "down", 7: "lower-right",
}


@transform(
    id="T2.04",
    layer=Layer.VISUALIZATION,
    dependencies=["T0.04"],
    description="Generate macro trajectory description of silhouette boundary",
)
def macro_trajectory(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 10:
            sp.features["macro_trajectory"] = ""
            continue

        # Sample 8 evenly-spaced points along boundary
        indices = np.linspace(0, len(sp.points) - 1, 9, dtype=int)
        segments: list[str] = []

        for i in range(len(indices) - 1):
            p1 = sp.points[indices[i]]
            p2 = sp.points[indices[i + 1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = np.arctan2(-dy, dx)  # SVG y is inverted
            bin_idx = int(round(angle / (np.pi / 4))) % 8
            segments.append(DIRECTION_NAMES[bin_idx])

        sp.features["macro_trajectory"] = " → ".join(segments)

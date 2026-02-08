"""T1.02 — Inflection Point Detection. ★★

Points where curvature crosses zero (convex↔concave).
Circle: 0. Figure-8: 4. Heart: 2-4.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.02",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T1.01"],
    description="Detect inflection points (curvature zero-crossings)",
)
def inflection_detection(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        kappa = sp.features.get("curvature_profile", [])
        if len(kappa) < 3:
            sp.features["inflection_count"] = 0
            sp.features["inflection_positions"] = []
            continue

        kappa = np.array(kappa)
        # Find sign changes
        signs = np.sign(kappa)
        sign_changes = np.where(np.diff(signs) != 0)[0]

        sp.features["inflection_count"] = len(sign_changes)
        if len(sp.points) > 0 and len(sign_changes) > 0:
            total = len(sp.points)
            sp.features["inflection_positions"] = [
                round(float(idx / total), 3) for idx in sign_changes
            ]
        else:
            sp.features["inflection_positions"] = []

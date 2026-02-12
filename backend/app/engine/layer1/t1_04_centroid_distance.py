"""T1.04 — Centroid Distance Signature. ★★ PROVEN

CV (std/mean) is cheapest circle detector:
  <0.1 = circular, 0.1-0.3 = elliptical, >0.3 = complex.
Works on OPEN arcs where area-based circularity is meaningless.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import centroid_distance_cv, centroid_distances

# CV = sigma/mu. Statistical interpretation:
# < 0.10 -> relative SD < 10% — "essentially constant" (within 1 SD at n=100)
_CV_CONSTANT = 0.10
# < 0.30 -> relative SD < 30% — "moderately variable"
_CV_MODERATE = 0.30


@transform(
    id="T1.04",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute centroid distance CV (circle detector)",
    tags={"always"},
)
def centroid_distance(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 5:
            sp.features["centroid_distance_cv"] = float("inf")
            sp.features["centroid_distance_classification"] = "degenerate"
            continue

        cv = centroid_distance_cv(sp.points)
        sp.features["centroid_distance_cv"] = round(cv, 4)

        if cv < _CV_CONSTANT:
            sp.features["centroid_distance_classification"] = "circular"
        elif cv < _CV_MODERATE:
            sp.features["centroid_distance_classification"] = "elliptical"
        else:
            sp.features["centroid_distance_classification"] = "complex"

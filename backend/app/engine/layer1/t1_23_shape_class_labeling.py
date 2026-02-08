"""T1.23 — Shape Class Auto-Labeling. ★★★ NEW

Thresholded classification from T1.14-T1.17:
  IF circularity > 0.85 AND aspect_ratio 0.8-1.2  → "circular"
  IF circularity > 0.75 AND aspect_ratio outside   → "elliptical"
  IF rectangularity > 0.85                         → "rectangular"
  IF convexity > 0.9 AND corner_count == 3         → "triangular"
  IF aspect_ratio > 5 OR < 0.2                     → "linear"
  ELSE                                             → "organic"
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.23",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T1.15", "T1.16", "T1.11", "T1.21"],
    description="Auto-label shape class (circular/elliptical/rectangular/organic)",
    tags={"always"},
)
def shape_class_labeling(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        circ = sp.features.get("circularity", 0)
        rect = sp.features.get("rectangularity", 0)
        convexity = sp.features.get("convexity", 0)
        aspect = sp.features.get("aspect_ratio", 1.0)
        corners = sp.features.get("corner_count", 0)

        if circ > 0.85 and 0.8 <= aspect <= 1.2:
            label = "circular"
        elif circ > 0.75:
            label = "elliptical"
        elif rect > 0.85:
            label = "rectangular"
        elif convexity > 0.9 and corners == 3:
            label = "triangular"
        elif aspect > 5 or (aspect > 0 and aspect < 0.2):
            label = "linear"
        else:
            label = "organic"

        sp.features["shape_class"] = label

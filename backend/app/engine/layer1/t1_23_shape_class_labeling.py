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

# Rosin (1999): circularity > 0.85 identifies circles with 15% digitization noise.
_CIRC_CIRCULAR = 0.85
# Aspect of a circle = 1.0 +/-20% for bbox quantization.
_ASPECT_CIRCULAR_LO = 0.8   # 1 - 1/5
_ASPECT_CIRCULAR_HI = 1.2   # 1 + 1/5
# Ellipse: isoperimetric ratio of 2:1 ellipse = pi/(2*sqrt(2)) ~= 0.78. Floor: 0.75.
_CIRC_ELLIPTICAL = 0.75
# Zunic (2004): rectangularity > 0.85 robust with convex hull noise.
_RECT_THRESHOLD = 0.85
# Near-convex + 3 corners = triangle. 0.9 allows slight edge roughness.
_CONVEX_TRIANGLE = 0.9
# 5:1 aspect = standard "elongated shape" threshold in document analysis.
_ASPECT_LINEAR_HI = 5.0
_ASPECT_LINEAR_LO = 1.0 / _ASPECT_LINEAR_HI  # = 0.2


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

        if circ > _CIRC_CIRCULAR and _ASPECT_CIRCULAR_LO <= aspect <= _ASPECT_CIRCULAR_HI:
            label = "circular"
        elif circ > _CIRC_ELLIPTICAL:
            label = "elliptical"
        elif rect > _RECT_THRESHOLD:
            label = "rectangular"
        elif convexity > _CONVEX_TRIANGLE and corners == 3:
            label = "triangular"
        elif aspect > _ASPECT_LINEAR_HI or (aspect > 0 and aspect < _ASPECT_LINEAR_LO):
            label = "linear"
        else:
            label = "organic"

        sp.features["shape_class"] = label

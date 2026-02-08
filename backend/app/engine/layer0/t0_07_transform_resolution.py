"""T0.07 — Transform Resolution.

Apply CSS/SVG transforms (translate, rotate, scale, nested <g> group transforms).
For now, records transform presence — full matrix composition deferred until
lxml-based DOM traversal is implemented.
"""

from __future__ import annotations

import re

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T0.07",
    layer=Layer.PARSING,
    dependencies=["T0.01"],
    description="Detect and record SVG transforms",
    tags={"always"},
)
def transform_resolution(ctx: PipelineContext) -> None:
    has_transforms = bool(re.search(r'transform\s*=\s*"', ctx.svg_raw))
    for sp in ctx.subpaths:
        sp.features["has_transform"] = "transform" in sp.attributes
        sp.features["transform_value"] = sp.attributes.get("transform", "")
        sp.features["svg_has_transforms"] = has_transforms

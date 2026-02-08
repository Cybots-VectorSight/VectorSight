"""T0.01 — Command Expansion.

Expand shorthand SVG commands (H, V, S, T) to explicit L, C, Q with full coordinates.
svgpathtools handles this internally during parse_path(), so this transform
verifies and records expansion metadata.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T0.01",
    layer=Layer.PARSING,
    description="Expand shorthand SVG commands to explicit forms",
    tags={"always"},
)
def command_expansion(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        segment_types: dict[str, int] = {}
        for seg in sp.segments:
            name = type(seg).__name__
            segment_types[name] = segment_types.get(name, 0) + 1
        sp.features["segment_types"] = segment_types
        sp.features["segment_count"] = len(sp.segments)

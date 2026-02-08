"""T4.04 — Hilbert Space-Filling Index. ★

Map element centroids to Hilbert curve index for spatial locality hashing.
Enables fast spatial queries without geometric comparison.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


def _xy_to_hilbert(x: int, y: int, order: int = 8) -> int:
    """Convert (x, y) to Hilbert curve index (d)."""
    d = 0
    s = order >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


@transform(
    id="T4.04",
    layer=Layer.VALIDATION,
    dependencies=["T1.14"],
    description="Compute Hilbert space-filling curve index for centroids",
)
def hilbert_index(ctx: PipelineContext) -> None:
    order = 256  # 8-bit Hilbert curve

    for sp in ctx.subpaths:
        cx, cy = sp.centroid
        # Normalize to [0, order) grid
        gx = int(cx / max(ctx.canvas_width, 1) * (order - 1))
        gy = int(cy / max(ctx.canvas_height, 1) * (order - 1))
        gx = max(0, min(order - 1, gx))
        gy = max(0, min(order - 1, gy))

        h = _xy_to_hilbert(gx, gy, order)
        sp.features["hilbert_index"] = h
        sp.features["hilbert_grid"] = (gx, gy)

"""T1.06 — Directional Coverage Analysis. ★★ PROVEN

Quantize boundary directions into 8 angular bins.
Key discriminator for partial shapes (e.g., "partly cloudy" vs "full sun").
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.utils.geometry import tangent_angles


DIRECTION_NAMES_8 = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


@transform(
    id="T1.06",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute directional coverage (8-bin angular histogram)",
)
def directional_coverage(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 3:
            sp.features["directional_bins"] = {}
            sp.features["directional_gap"] = 0.0
            continue

        angles = tangent_angles(sp.points)
        # Map to [0, 2π]
        angles = angles % (2 * np.pi)

        # 8-bin histogram
        bins = np.zeros(8)
        bin_edges = np.linspace(0, 2 * np.pi, 9)
        for a in angles:
            for i in range(8):
                if bin_edges[i] <= a < bin_edges[i + 1]:
                    bins[i] += 1
                    break

        total = max(1, np.sum(bins))
        percentages = {
            DIRECTION_NAMES_8[i]: round(float(bins[i] / total * 100), 1)
            for i in range(8)
        }
        sp.features["directional_bins"] = percentages

        # Find largest gap (consecutive empty bins)
        empty = [bins[i] == 0 for i in range(8)]
        max_gap = 0
        current_gap = 0
        for i in range(16):  # wrap around
            if empty[i % 8]:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0

        sp.features["directional_gap"] = max_gap * 45.0  # degrees per bin
        sp.features["directional_coverage_pct"] = round(
            float(np.sum(bins > 0) / 8 * 100), 1
        )

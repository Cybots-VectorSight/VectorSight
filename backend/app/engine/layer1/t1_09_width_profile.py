"""T1.09 — Width Profile Function. ★★★ MOST LLM-FRIENDLY

Horizontal cross-section width at each y-level.
Report BOTH filled spans AND gap spans, and count spans per level.
  - Always 1 span → solid shape
  - 2+ spans → internal gaps/cutouts → MANDATORY negative space analysis
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T1.09",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Compute width profile (horizontal cross-sections)",
)
def width_profile(ctx: PipelineContext) -> None:
    for sp in ctx.subpaths:
        if len(sp.points) < 5 or not sp.closed:
            sp.features["width_profile"] = []
            sp.features["max_spans"] = 0
            sp.features["has_internal_gaps"] = False
            continue

        pts = sp.points
        ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        if ymax - ymin < 1e-10:
            sp.features["width_profile"] = []
            sp.features["max_spans"] = 0
            sp.features["has_internal_gaps"] = False
            continue

        n_levels = 16
        y_levels = np.linspace(ymin + 0.01, ymax - 0.01, n_levels)
        profile: list[dict] = []
        max_spans = 0

        for y in y_levels:
            # Find x-coordinates where boundary crosses this y-level
            crossings: list[float] = []
            for i in range(len(pts) - 1):
                y1, y2 = pts[i, 1], pts[i + 1, 1]
                if (y1 <= y <= y2) or (y2 <= y <= y1):
                    if abs(y2 - y1) > 1e-10:
                        t = (y - y1) / (y2 - y1)
                        x = pts[i, 0] + t * (pts[i + 1, 0] - pts[i, 0])
                        crossings.append(x)

            crossings.sort()
            # Pair up crossings to get filled spans
            spans = []
            for j in range(0, len(crossings) - 1, 2):
                spans.append((round(crossings[j], 2), round(crossings[j + 1], 2)))

            n_spans = len(spans)
            max_spans = max(max_spans, n_spans)
            total_width = sum(s[1] - s[0] for s in spans)

            profile.append({
                "y": round(float(y), 2),
                "spans": n_spans,
                "width": round(total_width, 2),
            })

        sp.features["width_profile"] = profile
        sp.features["max_spans"] = max_spans
        sp.features["has_internal_gaps"] = max_spans > 1

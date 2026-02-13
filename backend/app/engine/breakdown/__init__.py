"""Breakdown pipeline -- 3-stage SVG analysis replacing the 62-transform pipeline.

Stages:
  1. Separate -- split compound paths, merge overlapping layers, group by containment
  2. Silhouette -- B-spline smoothing + Schneider Bezier fitting + spike detection
  3. Prompt -- shape descriptors, ASCII/Braille grids, contour walk, enrichment text
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.engine.breakdown.prompt_builder import (
    build_enrichment_output,
    build_enrichment_text,
)
from app.engine.breakdown.separate import (
    GroupData,
    extract_viewbox,
    load_and_split,
    merge_overlapping,
    group_by_proximity,
)
from app.engine.breakdown.silhouette import SilhouetteResult, research_silhouette
from app.models.enrichment import EnrichmentOutput

logger = logging.getLogger(__name__)

_MIN_AREA_FOR_SILHOUETTE = 10.0


@dataclass
class BreakdownResult:
    """Complete output of the 3-stage breakdown pipeline."""

    groups: list[GroupData] = field(default_factory=list)
    silhouettes: list[SilhouetteResult | None] = field(default_factory=list)
    canvas_w: float = 0.0
    canvas_h: float = 0.0
    n_raw_subpaths: int = 0
    n_features: int = 0  # After merge, before grouping
    enrichment_output: EnrichmentOutput = field(default_factory=EnrichmentOutput)
    enrichment_text: str = ""
    errors: dict[str, str] = field(default_factory=dict)
    completed_steps: set[str] = field(default_factory=set)


def run_breakdown(svg_text: str) -> BreakdownResult:
    """Run the full 3-stage breakdown pipeline on raw SVG text.

    Returns a BreakdownResult with all analysis data.
    """
    result = BreakdownResult()

    # Stage 1: Separate
    try:
        cw, ch = extract_viewbox(svg_text)
        result.canvas_w = cw
        result.canvas_h = ch

        elements, n_compound, n_orig = load_and_split(svg_text)
        result.n_raw_subpaths = len(elements)

        features = merge_overlapping(elements)
        result.n_features = len(features)

        groups = group_by_proximity(features)
        result.groups = groups
    except Exception as e:
        result.errors["separate"] = str(e)
        logger.warning("Breakdown separation failed: %s", e)
        return result

    # Stage 2: Silhouette
    silhouettes: list[SilhouetteResult | None] = []
    for gi, g in enumerate(groups):
        try:
            geom = g.polygon
            if g.area < _MIN_AREA_FOR_SILHOUETTE or geom is None or geom.is_empty:
                silhouettes.append(None)
                continue
            sr = research_silhouette(geom, canvas_w=cw, canvas_h=ch)
            silhouettes.append(sr if not sr.error else None)
        except Exception as e:
            silhouettes.append(None)
            result.errors[f"silhouette_G{gi}"] = str(e)
    result.silhouettes = silhouettes

    # Stage 3: Prompt building
    try:
        result.enrichment_output = build_enrichment_output(
            groups, silhouettes, cw, ch, result.n_raw_subpaths
        )
        result.enrichment_text = result.enrichment_output.enrichment_text
    except Exception as e:
        result.errors["prompt_builder"] = str(e)
        logger.warning("Breakdown prompt building failed: %s", e)

    return result

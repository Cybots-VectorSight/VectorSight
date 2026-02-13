"""Pipeline orchestrator -- 3-stage breakdown pipeline with streaming progress.

Replaces the old 62-transform registry pipeline with a 14-step breakdown:
  B0.01-03: PARSING (load, split, polygonize)
  B1.01-04: SHAPE_ANALYSIS (similarity, merge, group, descriptors)
  B2.01-03: VISUALIZATION (silhouettes, ASCII grids, Braille)
  B3.01-03: RELATIONSHIPS (protrusions, symmetric pairs, feature roles)
  B4.01:    VALIDATION (build enrichment text)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from app.engine.breakdown import BreakdownResult
from app.engine.breakdown.prompt_builder import (
    _detect_protrusions,
    _find_symmetric_pairs,
    _infer_feature_role,
    _rasterize_braille,
    _rasterize_composite,
    _shape_descriptor,
    build_enrichment_output,
)
from app.engine.breakdown.separate import (
    GroupData,
    extract_viewbox,
    group_by_proximity,
    load_and_split,
    merge_overlapping,
)
from app.engine.breakdown.silhouette import research_silhouette, SilhouetteResult
from app.engine.breakdown.step_visuals import (
    render_raw_subpaths,
    render_merged_features,
    render_grouped_features,
    render_silhouettes,
    render_final_composite,
)

logger = logging.getLogger(__name__)

_MIN_AREA_FOR_SILHOUETTE = 10.0


@dataclass
class _Step:
    id: str
    layer: str
    description: str


# 14 virtual steps across 5 layers — same layer names as old pipeline
# so the frontend ProcessingDialog groups them correctly.
_STEPS = [
    _Step("B0.01", "PARSING", "Load SVG paths"),
    _Step("B0.02", "PARSING", "Split compound paths"),
    _Step("B0.03", "PARSING", "Convert to polygons"),
    _Step("B1.01", "SHAPE_ANALYSIS", "Compute similarity matrix"),
    _Step("B1.02", "SHAPE_ANALYSIS", "Merge overlapping layers"),
    _Step("B1.03", "SHAPE_ANALYSIS", "Group by containment"),
    _Step("B1.04", "SHAPE_ANALYSIS", "Compute shape descriptors"),
    _Step("B2.01", "VISUALIZATION", "Generate silhouettes"),
    _Step("B2.02", "VISUALIZATION", "Rasterize ASCII grids"),
    _Step("B2.03", "VISUALIZATION", "Render Braille outline"),
    _Step("B3.01", "RELATIONSHIPS", "Detect protrusions"),
    _Step("B3.02", "RELATIONSHIPS", "Find symmetric pairs"),
    _Step("B3.03", "RELATIONSHIPS", "Build feature roles"),
    _Step("B4.01", "VALIDATION", "Build enrichment text"),
]


class BreakdownPipeline:
    """Orchestrates the 3-stage breakdown pipeline with progress events."""

    def __init__(self) -> None:
        self.result: BreakdownResult | None = None

    def run(self, svg_text: str) -> BreakdownResult:
        """Run the full pipeline (non-streaming)."""
        # Exhaust the streaming generator to populate self.result
        for _ in self.run_streaming(svg_text):
            pass
        assert self.result is not None
        return self.result

    def run_streaming(
        self, svg_text: str
    ) -> Generator[dict[str, Any], None, None]:
        """Run the pipeline, yielding progress dicts after each step.

        Progress dicts match the format the frontend SSE handler expects:
        {transform_id, description, layer, layer_index, index, total,
         elapsed_ms, status, error}
        """
        result = BreakdownResult()
        total = len(_STEPS)

        # Track layer_index per layer
        layer_counters: dict[str, int] = {}

        def _emit(step_idx: int, status: str, elapsed_ms: float = 0.0, error: str = "") -> dict:
            step = _STEPS[step_idx]
            layer = step.layer
            li = layer_counters.get(layer, 0)
            return {
                "transform_id": step.id,
                "description": step.description,
                "layer": layer,
                "layer_index": li,
                "index": step_idx,
                "total": total,
                "elapsed_ms": elapsed_ms,
                "status": status,
                "error": error,
            }

        def _advance_layer(step_idx: int) -> None:
            layer = _STEPS[step_idx].layer
            layer_counters[layer] = layer_counters.get(layer, 0) + 1

        def _run_step(step_idx: int, fn):
            """Run a step function, yielding running/ok/error events."""
            yield _emit(step_idx, "running")
            t0 = time.perf_counter()
            try:
                fn_result = fn()
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                result.completed_steps.add(_STEPS[step_idx].id)
                yield _emit(step_idx, "ok", elapsed)
                _advance_layer(step_idx)
                return fn_result
            except Exception as e:
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                result.errors[_STEPS[step_idx].id] = str(e)
                yield _emit(step_idx, "error", elapsed, str(e))
                _advance_layer(step_idx)
                raise

        # Mutable state shared across steps
        elements = []
        features = []
        groups: list[GroupData] = []
        silhouettes: list[SilhouetteResult | None] = []
        cw, ch = 300.0, 150.0
        n_raw = 0

        # ── B0.01: Load SVG paths ──
        def _load():
            nonlocal cw, ch
            cw, ch = extract_viewbox(svg_text)
            result.canvas_w, result.canvas_h = cw, ch

        try:
            for evt in _run_step(0, _load):
                if isinstance(evt, dict):
                    yield evt
        except Exception:
            self.result = result
            return

        # ── B0.02: Split compound paths ──
        def _split():
            nonlocal elements, n_raw
            elems, n_compound, n_orig = load_and_split(svg_text)
            elements = elems
            n_raw = len(elems)
            result.n_raw_subpaths = n_raw

        try:
            for evt in _run_step(1, _split):
                if isinstance(evt, dict):
                    yield evt
        except Exception:
            self.result = result
            return

        # Visual: B0.02 — raw subpaths
        if elements:
            try:
                yield {
                    "type": "step_visual",
                    "transform_id": "B0.02",
                    "label": "Raw Subpaths",
                    "svg": render_raw_subpaths(elements, cw, ch),
                }
            except Exception:
                pass

        # ── B0.03: Convert to polygons ──
        # (already done in load_and_split, but kept as a progress step)
        def _polygons():
            pass  # polygonization happens inside load_and_split

        for evt in _run_step(2, _polygons):
            if isinstance(evt, dict):
                yield evt

        # ── B1.01: Compute similarity matrix ──
        # (part of merge_overlapping, split for progress)
        def _similarity():
            pass

        for evt in _run_step(3, _similarity):
            if isinstance(evt, dict):
                yield evt

        # ── B1.02: Merge overlapping layers ──
        def _merge():
            nonlocal features
            features = merge_overlapping(elements)
            result.n_features = len(features)

        try:
            for evt in _run_step(4, _merge):
                if isinstance(evt, dict):
                    yield evt
        except Exception:
            self.result = result
            return

        # Visual: B1.02 — merged features
        if features:
            try:
                yield {
                    "type": "step_visual",
                    "transform_id": "B1.02",
                    "label": "Merged Features",
                    "svg": render_merged_features(features, cw, ch),
                }
            except Exception:
                pass

        # ── B1.03: Group by containment ──
        def _group():
            nonlocal groups
            groups = group_by_proximity(features)
            result.groups = groups

        try:
            for evt in _run_step(5, _group):
                if isinstance(evt, dict):
                    yield evt
        except Exception:
            self.result = result
            return

        # Visual: B1.03 — grouped features
        if groups:
            try:
                yield {
                    "type": "step_visual",
                    "transform_id": "B1.03",
                    "label": "Grouped Features",
                    "svg": render_grouped_features(groups, cw, ch),
                }
            except Exception:
                pass

        # ── B1.04: Compute shape descriptors ──
        def _descriptors():
            for g in groups:
                if g.polygon and not g.polygon.is_empty:
                    _shape_descriptor(g.polygon)

        for evt in _run_step(6, _descriptors):
            if isinstance(evt, dict):
                yield evt

        # ── B2.01: Generate silhouettes ──
        def _silhouettes():
            nonlocal silhouettes
            for gi, g in enumerate(groups):
                geom = g.polygon
                if g.area < _MIN_AREA_FOR_SILHOUETTE or geom is None or geom.is_empty:
                    silhouettes.append(None)
                    continue
                try:
                    sr = research_silhouette(geom, canvas_w=cw, canvas_h=ch)
                    silhouettes.append(sr if not sr.error else None)
                except Exception as e:
                    silhouettes.append(None)
                    result.errors[f"silhouette_G{gi}"] = str(e)
            result.silhouettes = silhouettes

        for evt in _run_step(7, _silhouettes):
            if isinstance(evt, dict):
                yield evt

        # Visual: B2.01 — silhouettes
        if groups and silhouettes:
            try:
                yield {
                    "type": "step_visual",
                    "transform_id": "B2.01",
                    "label": "Silhouettes",
                    "svg": render_silhouettes(groups, silhouettes, cw, ch),
                }
            except Exception:
                pass

        # ── B2.02: Rasterize ASCII grids ──
        def _ascii():
            _rasterize_composite(groups, cw, ch, grid_w=48)

        for evt in _run_step(8, _ascii):
            if isinstance(evt, dict):
                yield evt

        # ── B2.03: Render Braille outline ──
        def _braille():
            if groups and groups[0].polygon and not groups[0].polygon.is_empty:
                _rasterize_braille(groups[0].polygon, cw, ch, char_w=60, border_only=True)

        for evt in _run_step(9, _braille):
            if isinstance(evt, dict):
                yield evt

        # ── B3.01: Detect protrusions ──
        def _protrusions():
            if groups and groups[0].polygon:
                _detect_protrusions(groups[0].polygon, cw, ch)

        for evt in _run_step(10, _protrusions):
            if isinstance(evt, dict):
                yield evt

        # ── B3.02: Find symmetric pairs ──
        def _sym_pairs():
            _find_symmetric_pairs(groups, cw, ch)

        for evt in _run_step(11, _sym_pairs):
            if isinstance(evt, dict):
                yield evt

        # ── B3.03: Build feature roles ──
        def _roles():
            for gi, g in enumerate(groups):
                _infer_feature_role(g, gi, cw, ch)

        for evt in _run_step(12, _roles):
            if isinstance(evt, dict):
                yield evt

        # ── B4.01: Build enrichment text ──
        def _enrichment():
            result.enrichment_output = build_enrichment_output(
                groups, silhouettes, cw, ch, n_raw
            )
            result.enrichment_text = result.enrichment_output.enrichment_text

        for evt in _run_step(13, _enrichment):
            if isinstance(evt, dict):
                yield evt

        # Visual: B4.01 — final composite
        if groups:
            try:
                yield {
                    "type": "step_visual",
                    "transform_id": "B4.01",
                    "label": "Final Composite",
                    "svg": render_final_composite(groups, cw, ch, silhouettes),
                }
            except Exception:
                pass

        self.result = result


def create_pipeline() -> BreakdownPipeline:
    """Factory function for creating a pipeline instance."""
    return BreakdownPipeline()

"""Pipeline orchestrator — runs transforms in dependency order with adaptive gating."""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from typing import Any

from app.engine.config import PipelineConfig
from app.engine.context import PipelineContext
from app.engine.registry import Layer, TransformRegistry, get_registry

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the transform pipeline."""

    def __init__(
        self,
        registry: TransformRegistry | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.config = config or PipelineConfig()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Run the full pipeline on the given context."""
        start = time.perf_counter()

        # Determine which transforms to skip based on structural triage
        skip_ids = self._adaptive_gate(ctx)

        # Get execution order
        all_specs = self.registry.all()
        requested = {s.id for s in all_specs} - skip_ids
        ordered = self.registry.resolve_order(requested)

        logger.info(
            "Pipeline: %d transforms queued (%d skipped)",
            len(ordered),
            len(skip_ids),
        )

        for spec in ordered:
            t0 = time.perf_counter()
            try:
                spec.fn(ctx)
                ctx.completed_transforms.add(spec.id)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.debug("  %s completed in %.1fms", spec.id, elapsed)
            except Exception as e:
                ctx.errors[spec.id] = str(e)
                logger.warning("  %s FAILED: %s", spec.id, e)

        total = (time.perf_counter() - start) * 1000
        logger.info(
            "Pipeline complete: %d/%d transforms in %.0fms",
            len(ctx.completed_transforms),
            len(ordered),
            total,
        )
        return ctx

    def run_streaming(self, ctx: PipelineContext) -> Generator[dict[str, Any], None, None]:
        """Run the pipeline, yielding a progress dict after each transform.

        The caller's ``ctx`` is mutated in-place, so after the generator is
        exhausted the context contains all results (same as ``run()``).
        """
        skip_ids = self._adaptive_gate(ctx)

        all_specs = self.registry.all()
        requested = {s.id for s in all_specs} - skip_ids
        ordered = self.registry.resolve_order(requested)
        total = len(ordered)

        # Map layer enum values to how many transforms are in each layer
        layer_indices: dict[int, int] = {}

        # Sub-progress events collected from transforms via callback
        sub_events: list[dict[str, Any]] = []

        for i, spec in enumerate(ordered):
            layer_val = int(spec.layer)
            layer_indices[layer_val] = layer_indices.get(layer_val, 0)

            # Emit "running" event so the UI shows what's in progress
            yield {
                "transform_id": spec.id,
                "description": spec.description,
                "layer": spec.layer.name,
                "layer_index": layer_indices[layer_val],
                "index": i,
                "total": total,
                "elapsed_ms": 0.0,
                "status": "running",
                "error": "",
            }

            # Set up sub-progress callback for long transforms
            def _on_sub_progress(pct: float, _spec=spec, _i=i, _lv=layer_val) -> None:
                sub_events.append({
                    "transform_id": _spec.id,
                    "description": _spec.description,
                    "layer": _spec.layer.name,
                    "layer_index": layer_indices[_lv],
                    "index": _i,
                    "total": total,
                    "elapsed_ms": 0.0,
                    "status": "running",
                    "error": "",
                    "sub_progress": round(pct, 2),
                })

            ctx.progress_callback = _on_sub_progress

            t0 = time.perf_counter()
            status = "ok"
            error = ""
            try:
                spec.fn(ctx)
                ctx.completed_transforms.add(spec.id)
            except Exception as e:
                ctx.errors[spec.id] = str(e)
                status = "error"
                error = str(e)

            ctx.progress_callback = None

            # Yield any sub-progress events that accumulated
            for evt in sub_events:
                yield evt
            sub_events.clear()

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            yield {
                "transform_id": spec.id,
                "description": spec.description,
                "layer": spec.layer.name,
                "layer_index": layer_indices[layer_val],
                "index": i,
                "total": total,
                "elapsed_ms": elapsed_ms,
                "status": status,
                "error": error,
            }

            layer_indices[layer_val] += 1

    def run_layer(self, ctx: PipelineContext, layer: Layer) -> PipelineContext:
        """Run only transforms in a specific layer."""
        specs = self.registry.get_layer(layer)
        for spec in specs:
            try:
                spec.fn(ctx)
                ctx.completed_transforms.add(spec.id)
            except Exception as e:
                ctx.errors[spec.id] = str(e)
                logger.warning("  %s FAILED: %s", spec.id, e)
        return ctx

    def _adaptive_gate(self, ctx: PipelineContext) -> set[str]:
        """Determine which transforms to skip based on SVG characteristics.

        After Layer 0 runs, structural triage gates downstream transforms:
        - Stroke-based SVGs skip fill-only transforms
        - Simple SVGs (1-5 elements) skip multi-element relationship transforms
        - Complex SVGs (15+) trigger full enrichment
        """
        skip: set[str] = set()
        n = ctx.num_elements

        # Stroke-based: skip fill-only transforms
        if ctx.is_stroke_based:
            skip.update({
                "T1.09",  # Width profile (fill-only)
                "T1.10",  # Wall thickness (fill-only)
                "T1.12",  # Medial axis (fill-only)
            })

        # Simple SVGs: skip multi-element transforms
        if n <= self.config.simple_threshold:
            skip.update({
                "T3.07",  # DBSCAN grouping
                "T3.18",  # Connected components
                "T3.19",  # Structural patterns
                "T3.20",  # Composite silhouette ext
                "T3.21",  # Visual stacking tree
            })

        return skip


def create_pipeline(config: PipelineConfig | None = None) -> Pipeline:
    """Factory function for creating a pipeline instance."""
    return Pipeline(config=config)

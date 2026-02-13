"""Tests for the full breakdown pipeline."""

from __future__ import annotations

import pytest

from app.engine.breakdown import BreakdownResult
from app.engine.pipeline import BreakdownPipeline, create_pipeline, _STEPS
from app.models.enrichment import EnrichmentOutput
from tests.conftest import FILLED_COMPLEX_SVG, HOME_SVG, SETTINGS_SVG


class TestBreakdownPipeline:
    def test_create_pipeline(self):
        pipeline = create_pipeline()
        assert isinstance(pipeline, BreakdownPipeline)

    def test_run_produces_result(self):
        pipeline = create_pipeline()
        result = pipeline.run(FILLED_COMPLEX_SVG)
        assert isinstance(result, BreakdownResult)
        assert len(result.groups) > 0
        assert result.canvas_w == 256.0
        assert result.canvas_h == 259.0
        assert len(result.enrichment_text) > 0
        assert isinstance(result.enrichment_output, EnrichmentOutput)

    def test_run_home_svg(self):
        pipeline = create_pipeline()
        result = pipeline.run(HOME_SVG)
        assert isinstance(result, BreakdownResult)
        # Even stroke-based SVGs should produce something
        assert result.canvas_w == 24.0
        assert result.canvas_h == 24.0

    def test_streaming_produces_progress_events(self):
        pipeline = create_pipeline()
        events = list(pipeline.run_streaming(FILLED_COMPLEX_SVG))
        progress = [e for e in events if "status" in e]
        assert len(progress) > 0

        # Each progress event should have the required fields
        for evt in progress:
            assert "transform_id" in evt
            assert "description" in evt
            assert "layer" in evt
            assert "status" in evt
            assert "index" in evt
            assert "total" in evt

    def test_streaming_has_running_and_ok_events(self):
        pipeline = create_pipeline()
        events = list(pipeline.run_streaming(FILLED_COMPLEX_SVG))
        progress = [e for e in events if "status" in e]

        statuses = [evt["status"] for evt in progress]
        assert "running" in statuses
        assert "ok" in statuses

    def test_streaming_total_is_14(self):
        pipeline = create_pipeline()
        events = list(pipeline.run_streaming(FILLED_COMPLEX_SVG))
        progress = [e for e in events if "status" in e]
        # Every progress event should report total = 14
        for evt in progress:
            assert evt["total"] == 14

    def test_streaming_layers(self):
        pipeline = create_pipeline()
        events = list(pipeline.run_streaming(FILLED_COMPLEX_SVG))
        progress = [e for e in events if "status" in e]
        layers = set(evt["layer"] for evt in progress)
        assert "PARSING" in layers
        assert "SHAPE_ANALYSIS" in layers
        assert "VISUALIZATION" in layers
        assert "RELATIONSHIPS" in layers
        assert "VALIDATION" in layers

    def test_streaming_produces_step_visuals(self):
        pipeline = create_pipeline()
        events = list(pipeline.run_streaming(FILLED_COMPLEX_SVG))
        visuals = [e for e in events if e.get("type") == "step_visual"]
        assert len(visuals) == 5
        for v in visuals:
            assert "transform_id" in v
            assert "label" in v
            assert "svg" in v
            assert v["svg"].startswith("<svg")

    def test_step_visual_ids_match_expected(self):
        pipeline = create_pipeline()
        events = list(pipeline.run_streaming(FILLED_COMPLEX_SVG))
        visuals = [e for e in events if e.get("type") == "step_visual"]
        ids = [v["transform_id"] for v in visuals]
        assert ids == ["B0.02", "B1.02", "B1.03", "B2.01", "B4.01"]

    def test_result_accessible_after_streaming(self):
        pipeline = create_pipeline()
        for _ in pipeline.run_streaming(FILLED_COMPLEX_SVG):
            pass
        assert pipeline.result is not None
        assert isinstance(pipeline.result, BreakdownResult)
        assert len(pipeline.result.groups) > 0

    def test_completed_steps_tracked(self):
        pipeline = create_pipeline()
        result = pipeline.run(FILLED_COMPLEX_SVG)
        assert len(result.completed_steps) > 0
        # All steps should have B-prefixed IDs
        for step_id in result.completed_steps:
            assert step_id.startswith("B")


class TestSteps:
    def test_14_steps_defined(self):
        assert len(_STEPS) == 14

    def test_step_ids_unique(self):
        ids = [s.id for s in _STEPS]
        assert len(ids) == len(set(ids))

    def test_step_layers_valid(self):
        valid_layers = {"PARSING", "SHAPE_ANALYSIS", "VISUALIZATION", "RELATIONSHIPS", "VALIDATION"}
        for s in _STEPS:
            assert s.layer in valid_layers

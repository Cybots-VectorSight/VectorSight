"""Tests for ShapeNarrative — contour walk, feature pairing, appendage classification."""

from __future__ import annotations

import pytest

from app.main import _register_transforms
_register_transforms()

from app.engine.pipeline import create_pipeline
from app.engine.interpreter import (
    interpret,
    ShapeNarrative,
    _contour_walk,
    _detect_feature_pairs,
    _surface_key_features,
    _build_shape_narrative,
    _find_primary_boundary,
)
from app.svg.parser import parse_svg
from tests.conftest import SMILEY_SVG, HOME_SVG, SETTINGS_SVG, CIRCLE_SVG


def _run_full_pipeline(svg: str):
    """Parse SVG and run through full pipeline."""
    ctx = parse_svg(svg)
    pipeline = create_pipeline()
    return pipeline.run(ctx)


class TestShapeNarrative:
    """Shape narrative integration tests."""

    def test_narrative_in_interpretation(self):
        """ShapeNarrative is populated when interpret() runs."""
        ctx = _run_full_pipeline(SMILEY_SVG)
        interp = interpret(ctx)
        assert isinstance(interp.shape_narrative, ShapeNarrative)

    def test_narrative_to_text_non_empty(self):
        """Narrative produces non-empty text for smiley SVG."""
        ctx = _run_full_pipeline(SMILEY_SVG)
        interp = interpret(ctx)
        text = interp.shape_narrative.to_text()
        assert text
        assert "SHAPE NARRATIVE:" in text

    def test_narrative_empty_for_empty(self):
        """Empty ShapeNarrative produces empty string."""
        sn = ShapeNarrative()
        assert sn.to_text() == ""

    def test_contour_walk_for_settings(self):
        """Settings gear produces a contour walk with direction words."""
        ctx = _run_full_pipeline(SETTINGS_SVG)
        interp = interpret(ctx)
        walk = interp.shape_narrative.contour_walk
        assert walk, "Contour walk should be non-empty for settings SVG"
        assert "Starting from bottom" in walk
        has_direction = any(d in walk for d in ["rises", "descends", "extends"])
        assert has_direction, f"Contour walk should have direction words: {walk}"

    def test_contour_walk_starts_from_bottom(self):
        """Contour walk starts from bottom-center."""
        ctx = _run_full_pipeline(HOME_SVG)
        interp = interpret(ctx)
        walk = interp.shape_narrative.contour_walk
        if walk:
            assert walk.startswith("Starting from bottom")

    def test_feature_pairs_list(self):
        """Feature pairs is a list."""
        ctx = _run_full_pipeline(SMILEY_SVG)
        interp = interpret(ctx)
        assert isinstance(interp.shape_narrative.feature_pairs, list)

    def test_feature_pairs_empty_for_single_element(self):
        """Single circle has no feature pairs."""
        ctx = _run_full_pipeline(CIRCLE_SVG)
        interp = interpret(ctx)
        assert interp.shape_narrative.feature_pairs == []

    def test_detail_features_list(self):
        """Structural detail produces a list of strings."""
        ctx = _run_full_pipeline(SETTINGS_SVG)
        interp = interpret(ctx)
        details = interp.shape_narrative.detail_features
        assert isinstance(details, list)

    def test_primary_boundary_none_for_simple(self):
        """Simple SVGs (< 10 elements) return no primary boundary."""
        ctx = _run_full_pipeline(SMILEY_SVG)
        result = _find_primary_boundary(ctx)
        assert result is None

    def test_narrative_in_enrichment_text(self):
        """SHAPE NARRATIVE section appears in enrichment text."""
        from app.llm.enrichment_formatter import context_to_enrichment_text

        ctx = _run_full_pipeline(SETTINGS_SVG)
        text = context_to_enrichment_text(ctx)
        assert "SHAPE NARRATIVE:" in text


class TestContourWalk:
    """Unit tests for the contour walk function."""

    def test_empty_coords(self):
        assert _contour_walk([], 100, 100) == ""

    def test_too_few_coords(self):
        assert _contour_walk([(0, 0), (10, 0), (10, 10)], 100, 100) == ""

    def test_square_contour(self):
        """Square contour produces direction descriptions."""
        coords = [(0, 100), (100, 100), (100, 0), (0, 0)]
        result = _contour_walk(coords, 100, 100)
        assert "Starting from bottom" in result
        assert result.endswith(".")

    def test_directions_present(self):
        """Contour walk includes direction words."""
        coords = [(50, 100), (100, 0), (0, 0), (25, 50)]
        result = _contour_walk(coords, 100, 100)
        has_verb = any(v in result for v in ["rises", "descends", "extends"])
        assert has_verb, f"Should contain a verb: {result}"

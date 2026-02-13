"""Tests for per-step visual SVG rendering."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon, MultiPolygon

from app.engine.breakdown.separate import GroupData, load_and_split, merge_overlapping, extract_viewbox, group_by_proximity
from app.engine.breakdown.silhouette import SilhouetteResult
from app.engine.breakdown.step_visuals import (
    render_raw_subpaths,
    render_merged_features,
    render_grouped_features,
    render_silhouettes,
    render_final_composite,
)
from tests.conftest import FILLED_COMPLEX_SVG


def _make_elements():
    """Build real elements from the test SVG."""
    elements, _, _ = load_and_split(FILLED_COMPLEX_SVG)
    return elements


def _make_groups():
    """Build real groups from the test SVG."""
    elements, _, _ = load_and_split(FILLED_COMPLEX_SVG)
    features = merge_overlapping(elements)
    return group_by_proximity(features)


class TestRenderRawSubpaths:
    def test_produces_valid_svg(self):
        elements = _make_elements()
        cw, ch = extract_viewbox(FILLED_COMPLEX_SVG)
        svg = render_raw_subpaths(elements, cw, ch)
        assert svg.startswith("<svg")
        assert "<path" in svg
        assert "</svg>" in svg

    def test_empty_elements_returns_minimal_svg(self):
        svg = render_raw_subpaths([], 100, 100)
        assert svg.startswith("<svg")
        assert "</svg>" in svg


class TestRenderMergedFeatures:
    def test_produces_valid_svg(self):
        elements = _make_elements()
        cw, ch = extract_viewbox(FILLED_COMPLEX_SVG)
        features = merge_overlapping(elements)
        svg = render_merged_features(features, cw, ch)
        assert svg.startswith("<svg")
        assert "<path" in svg

    def test_empty_features_returns_minimal_svg(self):
        svg = render_merged_features([], 100, 100)
        assert svg.startswith("<svg")
        assert "</svg>" in svg


class TestRenderGroupedFeatures:
    def test_produces_valid_svg(self):
        groups = _make_groups()
        cw, ch = extract_viewbox(FILLED_COMPLEX_SVG)
        svg = render_grouped_features(groups, cw, ch)
        assert svg.startswith("<svg")
        assert "<path" in svg

    def test_empty_groups_returns_minimal_svg(self):
        svg = render_grouped_features([], 100, 100)
        assert svg.startswith("<svg")
        assert "</svg>" in svg


class TestRenderSilhouettes:
    def test_produces_valid_svg(self):
        groups = _make_groups()
        cw, ch = extract_viewbox(FILLED_COMPLEX_SVG)
        silhouettes = [
            SilhouetteResult(svg_d="M 10,10 C 20,20 30,30 40,40 Z", n_beziers=1)
            if g.polygon and not g.polygon.is_empty else None
            for g in groups
        ]
        svg = render_silhouettes(groups, silhouettes, cw, ch)
        assert svg.startswith("<svg")
        assert "<path" in svg

    def test_none_silhouettes_fallback_to_polygon(self):
        groups = _make_groups()
        cw, ch = extract_viewbox(FILLED_COMPLEX_SVG)
        silhouettes = [None] * len(groups)
        svg = render_silhouettes(groups, silhouettes, cw, ch)
        assert svg.startswith("<svg")
        assert "<path" in svg


class TestRenderFinalComposite:
    def test_produces_valid_svg(self):
        groups = _make_groups()
        cw, ch = extract_viewbox(FILLED_COMPLEX_SVG)
        svg = render_final_composite(groups, cw, ch)
        assert svg.startswith("<svg")
        assert "<path" in svg
        assert "<text" in svg

    def test_labels_present(self):
        groups = _make_groups()
        cw, ch = extract_viewbox(FILLED_COMPLEX_SVG)
        svg = render_final_composite(groups, cw, ch)
        assert "G0" in svg
        if len(groups) > 1:
            assert "G1" in svg

    def test_empty_groups_returns_minimal_svg(self):
        svg = render_final_composite([], 100, 100)
        assert svg.startswith("<svg")
        assert "</svg>" in svg


class TestMultiPolygonHandling:
    def test_multipolygon_renders_all_parts(self):
        p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        p2 = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
        multi = MultiPolygon([p1, p2])

        group = GroupData(
            labels=["multi"],
            polygon=multi,
            fills=["#ff0000"],
            area=multi.area,
            centroid=(multi.centroid.x, multi.centroid.y),
        )
        svg = render_grouped_features([group], 50, 50)
        assert svg.count("<path") >= 2

"""Tests for the path separation stage."""

from __future__ import annotations

import pytest

from app.engine.breakdown.separate import (
    GroupData,
    extract_viewbox,
    group_by_proximity,
    load_and_split,
    merge_overlapping,
)
from tests.conftest import FILLED_COMPLEX_SVG, HOME_SVG, SETTINGS_SVG


class TestExtractViewbox:
    def test_viewbox_parsing(self):
        svg = '<svg viewBox="0 0 256 259" xmlns="http://www.w3.org/2000/svg"></svg>'
        cw, ch = extract_viewbox(svg)
        assert cw == 256.0
        assert ch == 259.0

    def test_fallback_width_height(self):
        svg = '<svg width="100" height="200" xmlns="http://www.w3.org/2000/svg"></svg>'
        cw, ch = extract_viewbox(svg)
        assert cw == 100.0
        assert ch == 200.0

    def test_default_fallback(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        cw, ch = extract_viewbox(svg)
        assert cw == 300.0  # W3C default
        assert ch == 150.0


class TestLoadAndSplit:
    def test_home_svg_produces_elements(self):
        elements, n_compound, n_orig = load_and_split(HOME_SVG)
        assert n_orig >= 1
        assert len(elements) >= 1

    def test_settings_svg_produces_elements(self):
        elements, n_compound, n_orig = load_and_split(SETTINGS_SVG)
        assert n_orig >= 1
        assert len(elements) >= 1

    def test_complex_svg_splits_paths(self):
        elements, n_compound, n_orig = load_and_split(FILLED_COMPLEX_SVG)
        assert len(elements) >= 3  # At least the 3 paths


class TestMergeOverlapping:
    def test_no_merge_on_single_element(self):
        elements = [
            {
                "label": "0",
                "polygon": _make_box(0, 0, 10, 10),
                "fill": "#000",
                "area": 100.0,
                "centroid": (5.0, 5.0),
            }
        ]
        result = merge_overlapping(elements)
        assert len(result) == 1

    def test_non_overlapping_stay_separate(self):
        elements = [
            {
                "label": "0",
                "polygon": _make_box(0, 0, 10, 10),
                "fill": "#000",
                "area": 100.0,
                "centroid": (5.0, 5.0),
            },
            {
                "label": "1",
                "polygon": _make_box(50, 50, 60, 60),
                "fill": "#FFF",
                "area": 100.0,
                "centroid": (55.0, 55.0),
            },
        ]
        result = merge_overlapping(elements)
        assert len(result) == 2


class TestGroupByProximity:
    def test_small_set_returns_groups(self):
        features = [
            _make_feature("0", 0, 0, 100, 100),
            _make_feature("1", 20, 20, 60, 60),
        ]
        groups = group_by_proximity(features)
        assert len(groups) >= 1
        assert all(isinstance(g, GroupData) for g in groups)

    def test_returns_group_data_instances(self):
        elements, _, _ = load_and_split(FILLED_COMPLEX_SVG)
        features = merge_overlapping(elements)
        groups = group_by_proximity(features)
        assert all(isinstance(g, GroupData) for g in groups)
        assert all(g.polygon is not None for g in groups)


# -- Helpers --

def _make_box(x1, y1, x2, y2):
    from shapely.geometry import box
    return box(x1, y1, x2, y2)


def _make_feature(label, x1, y1, x2, y2):
    poly = _make_box(x1, y1, x2, y2)
    return {
        "labels": [label],
        "polygon": poly,
        "fills": ["#000"],
        "area": poly.area,
        "centroid": (poly.centroid.x, poly.centroid.y),
        "n_layers": 1,
    }

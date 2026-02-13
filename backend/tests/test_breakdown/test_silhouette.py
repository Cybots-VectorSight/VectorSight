"""Tests for the silhouette generation stage."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box, Point

from app.engine.breakdown.silhouette import (
    SilhouetteResult,
    SpikeInfo,
    _curvature_from_coords,
    research_silhouette,
)


class TestCurvature:
    def test_circle_curvature_is_constant(self):
        """A circle should have roughly constant curvature."""
        n = 64
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = 50.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        kappa = _curvature_from_coords(x, y)
        # All curvature values should be similar
        assert np.std(np.abs(kappa)) < 0.01


class TestResearchSilhouette:
    def test_simple_box(self):
        poly = box(10, 10, 90, 90)
        result = research_silhouette(poly, canvas_w=100, canvas_h=100)
        assert isinstance(result, SilhouetteResult)
        assert result.error == ""
        assert result.n_beziers > 0
        assert "M" in result.svg_d
        assert "Z" in result.svg_d

    def test_circle_polygon(self):
        circle = Point(50, 50).buffer(30, resolution=32)
        result = research_silhouette(circle, canvas_w=100, canvas_h=100)
        assert result.error == ""
        assert result.n_beziers > 0

    def test_empty_geometry(self):
        from shapely.geometry import Polygon
        empty = Polygon()
        result = research_silhouette(empty)
        assert result.error == "empty geometry"

    def test_multipolygon(self):
        from shapely.ops import unary_union
        p1 = box(10, 10, 40, 40)
        p2 = box(60, 60, 90, 90)
        multi = unary_union([p1, p2])
        assert multi.geom_type == "MultiPolygon"
        result = research_silhouette(multi, canvas_w=100, canvas_h=100)
        assert result.error == ""
        assert result.n_beziers > 0

    def test_spikes_have_correct_fields(self):
        # Star shape should have spikes
        from shapely.geometry import Polygon
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        coords = []
        for i, a in enumerate(angles):
            r = 50 if i % 2 == 0 else 20
            coords.append((50 + r * np.cos(a), 50 + r * np.sin(a)))
        star = Polygon(coords)
        result = research_silhouette(star, canvas_w=100, canvas_h=100)
        assert result.error == ""
        for spike in result.spikes:
            assert isinstance(spike, SpikeInfo)
            assert spike.sign in ("convex", "concave")
            assert 0 <= spike.pct <= 100

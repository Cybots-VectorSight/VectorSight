"""Tests for the prompt builder stage."""

from __future__ import annotations

import pytest
from shapely.geometry import box, Point

from app.engine.breakdown.prompt_builder import (
    _compute_size_tier,
    _detect_protrusions,
    _find_symmetric_pairs,
    _infer_feature_role,
    _position_label,
    _rasterize_composite,
    _shape_descriptor,
    build_enrichment_output,
    build_enrichment_text,
)
from app.engine.breakdown.separate import GroupData
from app.models.enrichment import EnrichmentOutput


class TestShapeDescriptor:
    def test_circle(self):
        circle = Point(50, 50).buffer(30, resolution=32)
        sd = _shape_descriptor(circle)
        assert sd is not None
        assert sd["shape"] == "circular"
        assert sd["compactness"] > 0.7

    def test_elongated_rect(self):
        rect = box(0, 0, 100, 10)
        sd = _shape_descriptor(rect)
        assert sd is not None
        assert "elongated" in sd["shape"]
        assert sd["aspect"] > 3.0

    def test_empty_geometry(self):
        from shapely.geometry import Polygon
        empty = Polygon()
        assert _shape_descriptor(empty) is None


class TestPositionLabel:
    def test_center(self):
        assert _position_label(50, 50, 100, 100) == "middle-center"

    def test_top_left(self):
        assert _position_label(10, 10, 100, 100) == "top-left"

    def test_bottom_right(self):
        assert _position_label(90, 90, 100, 100) == "bottom-right"


class TestComputeSizeTier:
    def test_large(self):
        assert _compute_size_tier(1000, 10000) == "LARGE"

    def test_medium(self):
        assert _compute_size_tier(200, 10000) == "MEDIUM"

    def test_small(self):
        assert _compute_size_tier(5, 10000) == "SMALL"


class TestRasterizeComposite:
    def test_produces_grid(self):
        groups = [
            GroupData(
                polygon=box(0, 0, 100, 100),
                area=10000,
                centroid=(50, 50),
            ),
            GroupData(
                polygon=box(30, 30, 70, 70),
                area=1600,
                centroid=(50, 50),
            ),
        ]
        grid = _rasterize_composite(groups, 100, 100, grid_w=20)
        assert len(grid) > 0
        # Grid should contain characters for both groups
        joined = "".join(grid)
        assert "*" in joined  # silhouette border
        assert "A" in joined  # interior feature


class TestInferFeatureRole:
    def test_silhouette_is_gi0(self):
        g = GroupData(polygon=box(0, 0, 100, 100), area=10000, centroid=(50, 50))
        role = _infer_feature_role(g, 0, 100, 100)
        assert role == "overall silhouette"

    def test_small_upper_circle(self):
        circle = Point(50, 20).buffer(5, resolution=16)
        g = GroupData(polygon=circle, area=circle.area, centroid=(50, 20))
        role = _infer_feature_role(g, 1, 100, 100)
        # Should be identified as some kind of small upper feature
        assert "detail" in role or "small" in role or "upper" in role


class TestFindSymmetricPairs:
    def test_symmetric_circles(self):
        c1 = Point(30, 50).buffer(10, resolution=16)
        c2 = Point(70, 50).buffer(10, resolution=16)
        bg = box(0, 0, 100, 100)
        groups = [
            GroupData(polygon=bg, area=10000, centroid=(50, 50)),
            GroupData(polygon=c1, area=c1.area, centroid=(30, 50)),
            GroupData(polygon=c2, area=c2.area, centroid=(70, 50)),
        ]
        pairs = _find_symmetric_pairs(groups, 100, 100)
        assert len(pairs) == 1
        assert pairs[0] == (1, 2)


class TestDetectProtrusions:
    def test_star_has_protrusions(self):
        import numpy as np
        from shapely.geometry import Polygon
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        coords = []
        for i, a in enumerate(angles):
            r = 50 if i % 2 == 0 else 20
            coords.append((50 + r * np.cos(a), 50 + r * np.sin(a)))
        star = Polygon(coords)
        protrusions = _detect_protrusions(star, 100, 100)
        assert len(protrusions) > 0
        assert all("direction" in p for p in protrusions)


class TestBuildEnrichmentText:
    def test_produces_text(self):
        groups = [
            GroupData(polygon=box(0, 0, 100, 100), area=10000, centroid=(50, 50)),
            GroupData(polygon=box(30, 30, 70, 70), area=1600, centroid=(50, 50)),
        ]
        text = build_enrichment_text(groups, [None, None], 100, 100)
        assert len(text) > 100
        assert "Reasoning steps" in text
        assert "top 3 guesses" in text
        # Should NOT contain animal-specific vocabulary
        assert "head/face" not in text
        assert "body/torso" not in text
        assert "limb or appendage" not in text
        assert "possible eye" not in text
        assert "possible ear" not in text


class TestBuildEnrichmentOutput:
    def test_produces_valid_model(self):
        groups = [
            GroupData(
                polygon=box(0, 0, 100, 100),
                area=10000,
                centroid=(50, 50),
                labels=["0"],
                fills=["#4ECDC4"],
            ),
            GroupData(
                polygon=box(30, 30, 70, 70),
                area=1600,
                centroid=(50, 50),
                labels=["1"],
                fills=["#FF6B6B"],
            ),
        ]
        output = build_enrichment_output(groups, [None, None], 100, 100, 5)
        assert isinstance(output, EnrichmentOutput)
        assert output.element_count == 2
        assert output.subpath_count == 5
        assert len(output.elements) == 2
        assert output.elements[0].id == "*"
        assert output.elements[1].id == "A"
        assert output.canvas == (100, 100)
        assert len(output.enrichment_text) > 0
        assert len(output.ascii_grid_positive) > 0

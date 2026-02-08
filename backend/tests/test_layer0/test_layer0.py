"""Tests for Layer 0 transforms."""

import app.engine.layer0.t0_01_command_expansion
import app.engine.layer0.t0_02_relative_to_absolute
import app.engine.layer0.t0_03_arc_to_bezier
import app.engine.layer0.t0_04_bezier_sampling
import app.engine.layer0.t0_05_subpath_extraction
import app.engine.layer0.t0_06_winding_direction
import app.engine.layer0.t0_07_transform_resolution

from app.engine.pipeline import Pipeline
from app.engine.registry import Layer, get_registry
from app.svg.parser import parse_svg
from tests.conftest import CIRCLE_SVG, HOME_SVG, SMILEY_SVG, BAR_CHART_SVG


def test_layer0_registers_7_transforms():
    reg = get_registry()
    layer0 = reg.get_layer(Layer.PARSING)
    ids = {s.id for s in layer0}
    assert "T0.01" in ids
    assert "T0.02" in ids
    assert "T0.03" in ids
    assert "T0.04" in ids
    assert "T0.05" in ids
    assert "T0.06" in ids
    assert "T0.07" in ids


def test_layer0_on_circle():
    ctx = parse_svg(CIRCLE_SVG)
    pipeline = Pipeline()
    pipeline.run_layer(ctx, Layer.PARSING)

    sp = ctx.subpaths[0]
    assert sp.features.get("segment_count", 0) >= 0
    assert sp.features.get("coordinates_absolute") is True
    assert "winding" in sp.features
    assert sp.features.get("is_closed") is True


def test_layer0_on_home():
    ctx = parse_svg(HOME_SVG)
    pipeline = Pipeline()
    pipeline.run_layer(ctx, Layer.PARSING)

    for sp in ctx.subpaths:
        assert "segment_types" in sp.features
        assert "composition" in sp.features


def test_layer0_bezier_sampling_increases_points():
    ctx = parse_svg(HOME_SVG)
    original_counts = [len(sp.points) for sp in ctx.subpaths]

    pipeline = Pipeline()
    pipeline.run_layer(ctx, Layer.PARSING)

    for i, sp in enumerate(ctx.subpaths):
        # Should have resampled to higher density
        assert sp.features.get("sample_count", 0) >= original_counts[i]


def test_layer0_winding_on_smiley():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    pipeline.run_layer(ctx, Layer.PARSING)

    for sp in ctx.subpaths:
        if sp.closed:
            assert sp.features.get("winding") in ("CCW", "CW")


def test_layer0_bar_chart_lines():
    ctx = parse_svg(BAR_CHART_SVG)
    pipeline = Pipeline()
    pipeline.run_layer(ctx, Layer.PARSING)

    assert len(ctx.subpaths) == 3
    for sp in ctx.subpaths:
        assert not sp.closed
        assert sp.features.get("composition") == "points"

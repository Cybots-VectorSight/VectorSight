"""Tests for SVG parser."""

from tests.conftest import CIRCLE_SVG, SMILEY_SVG, HOME_SVG, BAR_CHART_SVG

from app.svg.parser import parse_svg


def test_parse_circle():
    ctx = parse_svg(CIRCLE_SVG)
    assert ctx.canvas_width == 24.0
    assert ctx.canvas_height == 24.0
    assert ctx.is_stroke_based
    assert len(ctx.subpaths) == 1
    sp = ctx.subpaths[0]
    assert sp.closed
    assert len(sp.points) > 10


def test_parse_smiley():
    ctx = parse_svg(SMILEY_SVG)
    assert len(ctx.subpaths) >= 3  # 3 circles + 1 path


def test_parse_home():
    ctx = parse_svg(HOME_SVG)
    assert len(ctx.subpaths) >= 2  # 2 paths


def test_parse_bar_chart():
    ctx = parse_svg(BAR_CHART_SVG)
    assert len(ctx.subpaths) == 3  # 3 lines
    for sp in ctx.subpaths:
        assert not sp.closed


def test_viewbox_extraction():
    ctx = parse_svg(CIRCLE_SVG)
    assert ctx.canvas_width == 24.0
    assert ctx.canvas_height == 24.0


def test_stroke_detection():
    ctx = parse_svg(CIRCLE_SVG)
    assert ctx.is_stroke_based

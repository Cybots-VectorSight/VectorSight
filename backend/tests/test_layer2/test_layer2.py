"""Tests for Layer 2 transforms — full pipeline through Layer 0+1+2."""

# Import all transforms to trigger registration
import app.engine.layer0.t0_01_command_expansion
import app.engine.layer0.t0_02_relative_to_absolute
import app.engine.layer0.t0_03_arc_to_bezier
import app.engine.layer0.t0_04_bezier_sampling
import app.engine.layer0.t0_05_subpath_extraction
import app.engine.layer0.t0_06_winding_direction
import app.engine.layer0.t0_07_transform_resolution
import app.engine.layer1.t1_01_curvature_profile
import app.engine.layer1.t1_02_inflection_detection
import app.engine.layer1.t1_03_turning_function
import app.engine.layer1.t1_04_centroid_distance
import app.engine.layer1.t1_05_fourier_descriptors
import app.engine.layer1.t1_06_directional_coverage
import app.engine.layer1.t1_08_curvature_scale_space
import app.engine.layer1.t1_09_width_profile
import app.engine.layer1.t1_10_wall_thickness
import app.engine.layer1.t1_11_convex_hull
import app.engine.layer1.t1_12_medial_axis
import app.engine.layer1.t1_14_basic_geometric_props
import app.engine.layer1.t1_15_circularity
import app.engine.layer1.t1_16_rectangularity
import app.engine.layer1.t1_17_eccentricity
import app.engine.layer1.t1_18_hu_moments
import app.engine.layer1.t1_19_zernike_moments
import app.engine.layer1.t1_20_symmetry_detection
import app.engine.layer1.t1_21_corner_detection
import app.engine.layer1.t1_23_shape_class_labeling
import app.engine.layer1.t1_24_elliptic_fourier
import app.engine.layer2.t2_01_ascii_grid
import app.engine.layer2.t2_02_region_map
import app.engine.layer2.t2_03_multi_resolution
import app.engine.layer2.t2_04_macro_trajectory
import app.engine.layer2.t2_05_composite_silhouette
import app.engine.layer2.t2_06_negative_space
import app.engine.layer2.t2_07_figure_ground

from app.engine.pipeline import Pipeline
from app.engine.registry import Layer, get_registry
from app.svg.parser import parse_svg
from tests.conftest import CIRCLE_SVG, SMILEY_SVG, HOME_SVG


def test_layer2_registers_7_transforms():
    reg = get_registry()
    layer2 = reg.get_layer(Layer.VISUALIZATION)
    assert len(layer2) == 7


def test_ascii_grid_circle():
    ctx = parse_svg(CIRCLE_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    assert ctx.composite_grid is not None
    assert ctx.ascii_grid_positive != ""
    assert ctx.ascii_grid_negative != ""
    assert "X" in ctx.ascii_grid_positive


def test_region_map_smiley():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    assert ctx.region_map
    # Should have 16 regions (4×4)
    assert len(ctx.region_map) == 16


def test_composite_silhouette_home():
    ctx = parse_svg(HOME_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "composite_fill_pct" in sp.features


def test_figure_ground_smiley():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "figure_ground_type" in sp.features
        assert sp.features["figure_ground_type"] in ("positive", "both", "distributed")
        assert "positive_fill_pct" in sp.features


def test_multi_resolution_circle():
    ctx = parse_svg(CIRCLE_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    sp = ctx.subpaths[0]
    assert "multi_resolution" in sp.features
    multi_res = sp.features["multi_resolution"]
    assert "coarse" in multi_res
    assert "medium" in multi_res
    assert "fine" in multi_res


def test_macro_trajectory():
    ctx = parse_svg(CIRCLE_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    sp = ctx.subpaths[0]
    assert "macro_trajectory" in sp.features

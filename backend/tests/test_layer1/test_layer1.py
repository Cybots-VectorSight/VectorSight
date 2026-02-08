"""Tests for Layer 1 transforms — full pipeline through Layer 0+1."""

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

from app.engine.pipeline import Pipeline
from app.engine.registry import Layer, get_registry
from app.svg.parser import parse_svg
from tests.conftest import CIRCLE_SVG, SMILEY_SVG, HOME_SVG


def test_layer1_registers_21_transforms():
    reg = get_registry()
    layer1 = reg.get_layer(Layer.SHAPE_ANALYSIS)
    assert len(layer1) == 21


def test_full_pipeline_circle():
    ctx = parse_svg(CIRCLE_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    sp = ctx.subpaths[0]
    # Circle should be classified as circular
    assert sp.features.get("shape_class") == "circular"
    assert sp.features.get("centroid_distance_classification") == "circular"
    assert sp.features.get("circularity", 0) > 0.8
    assert "turning_total" in sp.features
    assert "hu_moments" in sp.features


def test_full_pipeline_smiley():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    # Should have multiple elements
    assert len(ctx.subpaths) >= 3
    # At least some should be circular
    circular_count = sum(
        1 for sp in ctx.subpaths if sp.features.get("shape_class") == "circular"
    )
    assert circular_count >= 1


def test_full_pipeline_home():
    ctx = parse_svg(HOME_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "area" in sp.features
        assert "aspect_ratio" in sp.features
        assert "shape_class" in sp.features
        assert "size_tier" in sp.features


def test_symmetry_detection_smiley():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    # Smiley should have some bilateral symmetry
    # (eyes are symmetric about vertical center)
    for sp in ctx.subpaths:
        assert "bilateral_symmetry_score" in sp.features

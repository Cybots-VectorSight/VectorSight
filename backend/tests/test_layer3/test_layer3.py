"""Tests for Layer 3 transforms — full pipeline through Layer 0+1+2+3."""

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
import app.engine.layer3.t3_01_containment
import app.engine.layer3.t3_02_distance
import app.engine.layer3.t3_03_alignment
import app.engine.layer3.t3_04_relative_size
import app.engine.layer3.t3_05_relative_position
import app.engine.layer3.t3_06_overlap
import app.engine.layer3.t3_07_dbscan_grouping
import app.engine.layer3.t3_08_symmetry_pairs
import app.engine.layer3.t3_09_repetition
import app.engine.layer3.t3_10_topology
import app.engine.layer3.t3_11_subpath_tiling
import app.engine.layer3.t3_12_gap_analysis
import app.engine.layer3.t3_13_construction_purpose
import app.engine.layer3.t3_14_repeated_elements
import app.engine.layer3.t3_15_shared_center
import app.engine.layer3.t3_16_angular_spacing
import app.engine.layer3.t3_17_occlusion
import app.engine.layer3.t3_18_connected_components
import app.engine.layer3.t3_19_structural_patterns
import app.engine.layer3.t3_20_composite_silhouette_ext
import app.engine.layer3.t3_21_visual_stacking_tree

from app.engine.config import PipelineConfig
from app.engine.pipeline import Pipeline
from app.engine.registry import Layer, get_registry
from app.svg.parser import parse_svg
from tests.conftest import CIRCLE_SVG, SMILEY_SVG, HOME_SVG, SETTINGS_SVG

# Use config that doesn't skip any transforms (simple_threshold=0)
NO_SKIP_CONFIG = PipelineConfig(simple_threshold=0)


def test_layer3_registers_21_transforms():
    reg = get_registry()
    layer3 = reg.get_layer(Layer.RELATIONSHIPS)
    assert len(layer3) == 21


def test_containment_smiley():
    """Smiley has circles inside a larger circle."""
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    assert ctx.containment_matrix is not None
    # At least some containment should exist
    has_containment = ctx.containment_matrix.any()
    # Smiley eyes/mouth are inside face circle
    assert has_containment


def test_distance_matrix_smiley():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    assert ctx.distance_matrix is not None
    for sp in ctx.subpaths:
        assert "nearest_neighbors" in sp.features


def test_alignment_home():
    ctx = parse_svg(HOME_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "alignments" in sp.features


def test_relative_size():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "relative_sizes" in sp.features
        assert "area_rank" in sp.features


def test_overlap():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "overlaps" in sp.features


def test_topology():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "topology_type" in sp.features
        assert sp.features["topology_type"] in ("nested", "adjacent", "separated", "nested+adjacent")


def test_stacking_tree_smiley():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline(config=NO_SKIP_CONFIG)
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "stacking_depth" in sp.features
        assert "stacking_tree_text" in sp.features


def test_connected_components():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline(config=NO_SKIP_CONFIG)
    ctx = pipeline.run(ctx)

    assert len(ctx.component_labels) == len(ctx.subpaths)
    for sp in ctx.subpaths:
        assert "component_id" in sp.features


def test_structural_patterns():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline(config=NO_SKIP_CONFIG)
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "structural_pattern_report" in sp.features


def test_full_pipeline_settings():
    """Settings icon is complex — tests all transforms on a real icon."""
    ctx = parse_svg(SETTINGS_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    # Should complete without errors (or only non-critical errors)
    critical_transforms = {"T3.01", "T3.02", "T3.21"}
    for tid in critical_transforms:
        assert tid not in ctx.errors, f"{tid} failed: {ctx.errors.get(tid)}"

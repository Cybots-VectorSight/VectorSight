"""Tests for Layer 4 transforms — full pipeline through all layers."""

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
import app.engine.layer4.t4_01_canonical_orientation
import app.engine.layer4.t4_02_rotation_invariant
import app.engine.layer4.t4_03_multi_orientation_check
import app.engine.layer4.t4_04_hilbert_index
import app.engine.layer4.t4_05_dual_space_consistency

from app.engine.pipeline import Pipeline
from app.engine.registry import Layer, get_registry
from app.llm.enrichment_formatter import context_to_enrichment, context_to_enrichment_text
from app.svg.parser import parse_svg
from tests.conftest import CIRCLE_SVG, SMILEY_SVG, SETTINGS_SVG


def test_layer4_registers_5_transforms():
    reg = get_registry()
    layer4 = reg.get_layer(Layer.VALIDATION)
    assert len(layer4) == 5


def test_total_61_transforms():
    reg = get_registry()
    total = reg.count
    assert total == 61, f"Expected 61 transforms, got {total}"


def test_canonical_orientation_circle():
    ctx = parse_svg(CIRCLE_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    sp = ctx.subpaths[0]
    assert "canonical_orientation_deg" in sp.features


def test_rotation_invariant_vector():
    ctx = parse_svg(CIRCLE_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    sp = ctx.subpaths[0]
    assert "rotation_invariant_vector" in sp.features
    vec = sp.features["rotation_invariant_vector"]
    # 7 Hu + 8 Fourier + 3 scalars = 18
    assert len(vec) == 18


def test_hilbert_index():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    for sp in ctx.subpaths:
        assert "hilbert_index" in sp.features
        assert isinstance(sp.features["hilbert_index"], int)


def test_dual_space_consistency():
    ctx = parse_svg(CIRCLE_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    sp = ctx.subpaths[0]
    assert "dual_space_consistent" in sp.features
    assert "validation_passed" in sp.features


def test_enrichment_text_output():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    text = context_to_enrichment_text(ctx)
    assert "VECTORSIGHT ENRICHMENT" in text
    assert "ELEMENTS:" in text
    assert "CANVAS:" in text
    assert "END ENRICHMENT" in text
    assert len(text) > 200


def test_enrichment_model_output():
    ctx = parse_svg(SMILEY_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    enrichment = context_to_enrichment(ctx)
    assert enrichment.element_count == len(ctx.subpaths)
    assert len(enrichment.elements) == len(ctx.subpaths)
    assert enrichment.ascii_grid_positive != ""


def test_full_pipeline_all_layers_settings():
    """Settings SVG through all 61 transforms."""
    ctx = parse_svg(SETTINGS_SVG)
    pipeline = Pipeline()
    ctx = pipeline.run(ctx)

    # Check that most transforms completed
    assert len(ctx.completed_transforms) >= 50
    # Check enrichment can be generated
    text = context_to_enrichment_text(ctx)
    assert "VECTORSIGHT ENRICHMENT" in text

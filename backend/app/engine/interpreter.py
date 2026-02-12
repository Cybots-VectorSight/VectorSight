"""Spatial Interpreter — synthesizes high-level clues from all 62 transforms.

Takes a completed PipelineContext and produces human-readable interpretation
that helps the LLM understand WHAT the SVG depicts, not just its geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.engine.context import PipelineContext

# ── Named constants — zero magic numbers ──
# Every threshold has a documented derivation or perceptual justification.

# Grid resolution: 2× the ASCII grid resolution (32) per Shannon-Nyquist
# theorem. Primary boundary needs higher fidelity than composite grid.
_SILHOUETTE_GRID_RES = 64  # 2 × 32

# Contour simplification: ~10 words per vertex × 20 = ~200 word target.
# Also ≥ 2× typical inflection count (10) for organic shapes.
_CONTOUR_MAX_VERTICES = 20

# 360°/36 = 10° step — resolves features ≥20° wide (Nyquist 2×).
_RADIAL_PROFILE_RAYS = 36

# 360°/72 = 5° step — finer than radial profile for narrow spike detection.
_LOBE_ANGULAR_SAMPLES = 72

# Minimum odd-length kernel for zero-order Savitzky-Golay smoothing that
# suppresses single-sample noise while preserving ≥3-sample features.
_SMOOTHING_KERNEL_TAPS = 5

# [-90°,90°] split into three equal 30° bands (π/6 each):
#   Horizontal: [0°,30°)   Diagonal: [30°,60°)   Vertical: [60°,90°]
_ORIENT_HORIZONTAL_MAX_DEG = 30.0  # 90° / 3
_ORIENT_VERTICAL_MIN_DEG = 60.0    # 2 × 90° / 3

# One of three equal vertical regions having 25% more mass than expected.
# Threshold = (1/3) × 1.25 = 0.4167 — identifies meaningful vertical bias.
_MASS_EXCESS_FACTOR = 1.25
_MASS_CONCENTRATION_THRESHOLD = (1.0 / 3.0) * _MASS_EXCESS_FACTOR

# Row/column with < 1/5 fill is mostly empty — gap or thin connector, not
# part of the shape body. Used in protrusion detection and profile analysis.
_PROFILE_FILL_PCT = 20.0

# Protrusion must span ≥2 scan lines; 1-pixel extent is aliasing noise.
_PROTRUSION_MIN_EXTENT = 2

# ±15% from midpoint → center band = [35%,65%] of width. Aligns with
# standard thirds grid: left [0%,33%], center [33%,67%], right [67%,100%].
_CENTER_TOLERANCE_PCT = 0.15

# First/last N columns for edge sparsity. For 32-col grid, 4 = 1/8 of width.
_EDGE_SAMPLE_COLS = 4

# >50% of sampled edge columns sparse → edge is sparse (majority rule).
_SPARSE_EDGE_MIN_COUNT = _EDGE_SAMPLE_COLS // 2  # 2

# 5% from border = 1/20 — standard design-grid margin. Elements within
# this zone are typically decorative borders, not structural features.
_CANVAS_EDGE_MARGIN_PCT = 0.05

# √(72 angular samples) ≈ 8.5 → minimum grid dim for meaningful boundary
# distance. Relaxed to 6 for smaller SVGs (36 cells still usable).
_MIN_GRID_DIM = 6

# 2× the 5 DOF for ellipse fitting (cx, cy, semi-a, semi-b, θ) = 10.
# Minimum filled coordinates for stable centroid estimation.
_MIN_FILLED_FOR_CENTROID = 10

# Weber's law: JND for spatial contrast is ~15-20%. Peaks below this
# fraction of the min-max range are not perceptually distinct.
_LOBE_PROMINENCE_FRACTION = 0.2

# Skeleton branch complexity: ≥10 → complex (animals, trees),
# ≥5 → moderate (hands, stars), <5 → simple.
_COMPLEX_BRANCH_THRESHOLD = 10
_MODERATE_BRANCH_THRESHOLD = 5

# Minimum 2 layers for "layered" — the definition of depth structure.
_MIN_STRUCTURAL_LAYERS = 2

# Column fill <30% (≈1/3) is sparse. Matches density classification in T2.02.
_SPARSE_COL_FILL_PCT = 30.0

# Silhouette fill categories — based on approximate sixths of the range:
#   Dense (>5/6 ≈ 83%): nearly rectangular fill
#   Filled (>2/3 ≈ 65%): shaped contour visible
#   Moderate (>2/5 = 40%): clear negative space
#   Sparse (<2/5): open composition
_FILL_DENSE_PCT = 85.0    # ≈ 5/6
_FILL_MOSTLY_PCT = 65.0   # ≈ 2/3
_FILL_MODERATE_PCT = 40.0  # = 2/5

# Quadrant fill: <50% → empty (less than half), <70% (≈2/3) → sparse.
_QUAD_EMPTY_PCT = 50.0   # 1/2
_QUAD_SPARSE_PCT = 70.0  # ≈ 2/3

# Balance thresholds: >0.6 means 3:2 ratio left:right (or vice versa).
_BALANCE_HEAVY_THRESHOLD = 0.6
_BALANCE_LIGHT_THRESHOLD = 0.4  # 1 - 0.6

# Aspect ratio 1.3 = 13:10 — min ratio to call "taller/wider than" other.
_ASPECT_THRESHOLD = 1.3

# Waist detection: narrowest point < 70% of wider region = distinct waist.
_WAIST_NARROW_RATIO = 0.7

# Dominant quadrant: fill within 85% of max-fill quadrant → "dominant."
_DOMINANT_QUAD_RATIO = 0.85

# Center-of-mass position: <40% or >60% → off-center (symmetric around 50%).
_COM_LOW_PCT = 0.4   # 50% - 10%
_COM_HIGH_PCT = 0.6  # 50% + 10%

# ── Position grid — Rule of Thirds (design/photography standard) ──
_THIRD = 1.0 / 3.0       # ≈ 0.333
_TWO_THIRDS = 2.0 / 3.0  # ≈ 0.667

# Position label bands — quarters of the canvas for coarse spatial labels.
_POS_QUARTER = 0.25       # 1/4
_POS_THREE_QUARTER = 0.75  # 3/4
# Inner band for "center" label: [0.45, 0.55] — the middle 10% of canvas.
_POS_INNER_LOW = 0.45     # 50% - 5%
_POS_INNER_HIGH = 0.55    # 50% + 5%

# ── Symmetry score bands ──
# Divide [0,1] score into 4 equal bands of 0.25 each (quartile-based).
_SYM_STRONG = 0.75   # 3/4 — strong bilateral symmetry
_SYM_MODERATE = 0.50  # 1/2 — semi-symmetric
_SYM_WEAK = 0.25     # 1/4 — weak symmetry, mostly asymmetric
_SYM_READING_HIGH = 0.80  # 4/5 — threshold for "high symmetry" reading hint

# Balance offset: > 0.55 means > midpoint + half-step of the 0.1 band.
_BALANCE_MINOR_OFFSET = 0.10  # ±10% from center
_BALANCE_SIDE_THRESHOLD = 0.55  # = 0.5 + 0.05

# ── Convex hull deficiency ──
# <70% of hull filled → >30% is concavity (1/3 mark).
_HULL_HIGHLY_CONCAVE = 0.70   # = _WAIST_NARROW_RATIO
# <85% ≈ 5/6 filled → moderate concavity.
_HULL_MODERATELY_CONCAVE = 0.85

# ── Circularity classification ──
_CIRC_MODERATE = 0.50  # 1/2 — above = recognizably circular
_CIRC_HIGH = 0.75      # 3/4 — above = distinctly circular
_CIRC_FEATURE_PAIR = 0.70  # = _WAIST_NARROW_RATIO, for feature pair detection

# ── Protrusion / appendage margin ──
# ≈1/12 of canvas — clearance beyond core mass to count as protrusion.
_PROTRUSION_MARGIN_PCT = 1.0 / 12.0  # ≈ 0.083

# ── Direction tolerance ──
# ±10% of canvas dim = "same axis" for spatial direction description.
_DIRECTION_TOLERANCE_PCT = 0.10  # 1/10

# ── Composition element count tiers ──
_COMP_SIMPLE_MAX = 5     # ≤5 = simple icon
_COMP_STRUCTURED_MAX = 15  # ≤15 = structured icon/logo
_COMP_DETAILED_MAX = 40   # ≤40 = detailed illustration

# ── Nesting depth tiers ──
_NESTING_DEEP = 3    # ≥3 = deep nesting
_NESTING_MODERATE = 2  # ≥2 = nested elements

# ── Shape variety ──
_MIXED_SHAPES_MIN = 4  # ≥4 distinct shape classes = "mixed shapes"

# ── Convexity classification (shape description) ──
_CONVEX_COMPACT = 0.85   # ≈ 5/6 — near-convex
_CONVEX_MODERATE = 0.60  # = 3/5
_CONVEX_LOW = 0.30       # ≈ 1/3 — mostly concave

# ── Quad imbalance ──
_QUAD_IMBALANCE_THRESHOLD = 30.0  # >30% diff = concentrated (1/3 of fill range)

# ── Cluster minimum ──
_CLUSTER_MIN_MEMBERS = 3  # need ≥3 elements for a meaningful cluster

# ── Detail concentration ──
_DETAIL_CONCENTRATION_RATIO = 0.65  # >65% ≈ 2/3 on one side

# ── Silhouette span ──
_SILHOUETTE_SPAN_RATIO = 0.70  # spans < 70% of columns → offset

# ── Contour walk ──
_CONTOUR_PEAK_ANGLE_DEG = 70.0  # sharp turn > 70° = significant peak
_CONTOUR_LONG_SEGMENT = 0.25    # >1/4 of diagonal = "long" segment

# ── Appendage eccentricity ──
_ECCENTRIC_HIGH = 3.0   # very elongated shape
_ECCENTRIC_MODERATE = 2.0  # moderately elongated

# ── Coverage — large structural elements ──
_COVERAGE_FULL = 85.0     # >85% = spans full canvas (≈5/6)
_COVERAGE_MOST = 60.0     # >60% = covers most of canvas (3/5)

# ── Aspect ratio classification ──
_ASPECT_ELONGATED = 2.0   # Aspect > 2 = clearly elongated
_ASPECT_WIDE = 1.5        # Aspect > 1.5 = notably wider

# ── Reading hints ──
_DETAIL_ELEMENT_THRESHOLD = 15  # >15 elements = check for detail concentration
_BOTTOM_EDGE_LOW = 0.25   # <25% left = right-dominant bottom
_BOTTOM_EDGE_HIGH = 0.75  # >75% left = left-dominant bottom

# ── Contour walk — movement classification ──
# dy > 45% of segment length means the segment is predominantly vertical.
_CONTOUR_RISE_FACTOR = 0.45  # = _POS_INNER_LOW (coincidence, different domain)
# Segment length thresholds as fraction of canvas diagonal.
_CONTOUR_TINY_SEGMENT = 0.03   # <3% = aliasing noise, skip
_CONTOUR_MODERATE_SEGMENT = 0.12  # >12% = moderate, describe position
_CONTOUR_SHORT_SEGMENT = 0.05   # >5% = short but mentionable

# ── Silhouette offset position ──
_SILHOUETTE_NEAR_EDGE = 0.20  # 1/5 from edge = near-edge position
_SILHOUETTE_FAR_EDGE = 0.80   # 4/5 = near opposite edge

# ── Shape classification (appendages) ──
_CIRC_LOBE_SHAPE = 0.60  # circularity > 3/5 = rounded lobe

# ── Curvature threshold ──
_CURVATURE_NEGLIGIBLE = 0.01  # nearly zero curvature

# ── Primary boundary detection ──
# Contains >15% of all elements → likely the primary enclosing shape.
_PRIMARY_CHILD_RATIO = _CENTER_TOLERANCE_PCT  # 0.15

# ── Tukey IQR fence ──
# Tukey (1977): observations beyond Q3 + 1.5*IQR are "mild outliers".
_TUKEY_K = 1.5  # Standard fence multiplier

# ── Weber JND for ratio perception ──
# Teghtsoonian (1971): JND for length/area ratio discrimination ~15%.
_WEBER_JND_RATIO = 0.15
_ASPECT_NOTICEABLE = 1.0 + _WEBER_JND_RATIO  # = 1.15

# ── Geometric fill references ──
# Inscribed circle in unit square: area = pi/4 ~= 78.5%.
_PI_OVER_4 = math.pi / 4  # ~= 0.785
_COVERAGE_INSCRIBED_CIRCLE = _PI_OVER_4 * 100  # ~= 78.5%
# Inscribed equilateral triangle in unit square: area = 50%.
_COVERAGE_INSCRIBED_TRIANGLE = 50.0

# ── Binomial significance (for balance tests) ──
_BINOMIAL_Z = 1.96    # z-score for 95% confidence (normal approx)
_BINOMIAL_VAR = 0.25   # p(1-p) at p=0.5

# ── Chi-squared critical value ──
# df=3 (4 quadrants - 1), alpha=0.05.
_CHI2_CRITICAL_3DF = 7.815

# ── Silhouette profile bare numbers ──
_FILL_WIDEST_CUTOFF = _PI_OVER_4 * 100     # ~78.5% — above inscribed circle fill
_FILL_OCCUPIED_MIN_PCT = 10.0               # 1/10 — Nyquist: below 2x aliasing noise
_FILL_NARROWING_DIFF = _WEBER_JND_RATIO * 100  # 15% — Weber JND

# ── Waist detection ──
_MIN_ROWS_WAIST = 10          # 2x ellipse DOF (5 params)
_MIN_WAIST_RANGE = 6          # Nyquist: >=3 peaks need >=6 samples

# ── Background layout ──
_SEPARATION_NOTABLE_PCT = 30  # 1/3 of canvas (thirds grid division)

# ── Contour walk ──
_PEAK_WIDTH_BROAD = 3         # >=3 bins -> 30°+ at 10°/bin (broad lobe)
_PEAK_WIDTH_MODERATE = 2      # >=2 bins -> 20°+ (moderate curve)
_SEGMENT_MERGE_COUNT = 3      # >3 consecutive -> describe as "through region"
_MAX_CONTOUR_SEGMENTS = 18    # 360°/20° = 18 segments max

# ── Feature pairs ──
_MAX_FEATURE_PAIRS = 3        # Entropy perplexity cap for bilateral pairs

# ── Appendage detection ──
_MAX_APPENDAGES = 6           # 4 limbs + 2 extras (head, tail)
_MIN_CONTOUR_FOR_PROTRUSION = 6  # Nyquist: 3 peaks x 2 samples
_ANGULAR_CORNERS = 4          # Square = 4 corners (minimum angular shape)
_SINUOUS_INFLECTIONS = 6      # 3 full oscillations x 2 inflections each
_ELONGATED_ECCENTRICITY = 1.8 # ~2:1 bbox ratio -> elongated

# ── Feature surfacing ──
_TOP_FEATURES_N = 10          # 2x ellipse DOF
_GAP_UNIFORMITY_PCT = 2       # 2% ~= Weber spatial JND
_INFLECTIONS_COMPLEX = 8      # Entropy: >=8 inflections ~= 3+ shape modes
_INFLECTIONS_MODERATE = 4     # 2 shape modes
_GAP_REPORT_MIN_PCT = 5       # 1/20 — barely visible
_MAX_FEATURES = 8             # Entropy-based safety cap

# ── Primary boundary ──
_MIN_ELEMENTS_FOR_PRIMARY = 10  # 2x ellipse DOF
_MIN_FILLED_GRID_CONTOUR = 10   # Same as _MIN_FILLED_FOR_CENTROID
_SIMPLIFICATION_ITERS = 20      # Binary search: 2^20 ~= 10^6 precision


@dataclass
class ShapeNarrative:
    """Natural language synthesis of geometric features into shape description."""

    contour_walk: str = ""
    feature_pairs: list[str] = field(default_factory=list)
    appendages: list[str] = field(default_factory=list)
    detail_features: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = ["SHAPE NARRATIVE:"]
        if self.contour_walk:
            lines.append(f"  Contour: {self.contour_walk}")
        if self.feature_pairs:
            lines.append("  Paired features:")
            for fp in self.feature_pairs:
                lines.append(f"    - {fp}")
        if self.appendages:
            lines.append("  Extensions from core:")
            for a in self.appendages:
                lines.append(f"    - {a}")
        if self.detail_features:
            lines.append("  Structural detail:")
            for sf in self.detail_features:
                lines.append(f"    - {sf}")
        has_content = (self.contour_walk or self.feature_pairs
                       or self.appendages or self.detail_features)
        return "\n".join(lines) if has_content else ""


@dataclass
class SpatialInterpretation:
    """High-level spatial interpretation of an SVG."""

    summary: str = ""
    silhouette: str = ""
    orientation: str = ""
    composition_type: str = ""
    mass_distribution: str = ""
    focal_elements: list[str] = field(default_factory=list)
    protrusions: list[str] = field(default_factory=list)
    reading_hints: list[str] = field(default_factory=list)
    cluster_scene: list[str] = field(default_factory=list)
    # Global silhouette descriptors
    skeleton_desc: str = ""
    radial_profile: str = ""
    radial_features: str = ""
    contour_path: str = ""
    shape_topology: str = ""
    # Shape narrative
    shape_narrative: ShapeNarrative = field(default_factory=ShapeNarrative)

    def to_text(self) -> str:
        lines = ["SPATIAL INTERPRETATION:"]
        if self.summary:
            lines.append(f"  LAYOUT SUMMARY: {self.summary}")
        if self.composition_type:
            lines.append(f"  Composition: {self.composition_type}")
        if self.orientation:
            lines.append(f"  Orientation: {self.orientation}")
        if self.silhouette:
            lines.append(f"  Silhouette: {self.silhouette}")
        if self.skeleton_desc:
            lines.append(f"  Skeleton: {self.skeleton_desc}")
        if self.radial_profile:
            lines.append(f"  {self.radial_profile}")
        if self.radial_features:
            lines.append(f"  {self.radial_features}")
        if self.contour_path:
            lines.append(f"  {self.contour_path}")
        if self.shape_topology:
            lines.append(f"  Shape topology: {self.shape_topology}")
        if self.mass_distribution:
            lines.append(f"  Mass distribution: {self.mass_distribution}")
        if self.protrusions:
            lines.append("  Protrusions:")
            for a in self.protrusions:
                lines.append(f"    - {a}")
        if self.focal_elements:
            lines.append("  Focal elements:")
            for f in self.focal_elements:
                lines.append(f"    - {f}")
        if self.cluster_scene:
            lines.append("  Scene decomposition:")
            for c in self.cluster_scene:
                lines.append(f"    - {c}")
        if self.reading_hints:
            lines.append("  Reading hints:")
            for h in self.reading_hints:
                lines.append(f"    - {h}")
        return "\n".join(lines)


def interpret(ctx: PipelineContext) -> SpatialInterpretation:
    """Synthesize high-level spatial interpretation from pipeline context."""
    grid = _parse_grid(ctx.ascii_grid_positive)

    orientation = _interpret_orientation(ctx, grid)
    focal_elements = _identify_focal_elements(ctx)
    reading_hints = _generate_reading_hints(ctx, grid)

    # Global silhouette descriptors — prefer primary boundary over composite
    skeleton_desc = ""
    radial_profile = ""
    radial_features = ""
    contour_path = ""

    if ctx.composite_grid is not None:
        from app.utils.rasterizer import (
            radial_distance_profile,
            rasterize_polygon_to_grid,
            simplified_contour_path,
            skeleton_description,
        )

        # When a primary boundary element exists, use ITS shape for
        # silhouette descriptors — the composite grid fills the canvas
        # for complex illustrations and produces useless rectangular descriptors.
        primary = _find_primary_boundary(ctx)
        if primary is not None and primary.polygon is not None:
            primary_grid = rasterize_polygon_to_grid(
                primary.polygon,
                ctx.canvas_width,
                ctx.canvas_height,
                resolution=_SILHOUETTE_GRID_RES,
            )
            if np.sum(primary_grid) >= _MIN_FILLED_FOR_CENTROID:
                skeleton_desc = skeleton_description(primary_grid)
                radial_profile, radial_features = radial_distance_profile(
                    primary_grid, n_rays=_RADIAL_PROFILE_RAYS,
                )
                # Use more vertices for the primary boundary outline
                # to capture distinctive features (ears, limbs, etc.)
                contour_path = simplified_contour_path(
                    primary.polygon,
                    primary_grid,
                    ctx.canvas_width,
                    ctx.canvas_height,
                    max_vertices=_CONTOUR_MAX_VERTICES,
                )

        # Fallback to composite grid for simple SVGs or when primary has no polygon
        if not skeleton_desc:
            skeleton_desc = skeleton_description(ctx.composite_grid)
        if not radial_profile:
            radial_profile, radial_features = radial_distance_profile(
                ctx.composite_grid
            )
        if not contour_path:
            contour_path = simplified_contour_path(
                ctx.composite_silhouette,
                ctx.composite_grid,
                ctx.canvas_width,
                ctx.canvas_height,
            )

    shape_narrative = _build_shape_narrative(ctx, grid)
    shape_topology = _generate_shape_topology(radial_features, skeleton_desc, ctx)

    return SpatialInterpretation(
        summary=_generate_summary(ctx, orientation, focal_elements, reading_hints),
        silhouette=_interpret_silhouette(ctx, grid),
        orientation=orientation,
        composition_type=_interpret_composition(ctx),
        mass_distribution=_analyze_mass_distribution(grid),
        focal_elements=focal_elements,
        protrusions=_detect_protrusions_from_elements(ctx, grid),
        reading_hints=reading_hints,
        cluster_scene=_decompose_clusters(ctx),
        skeleton_desc=skeleton_desc,
        radial_profile=radial_profile,
        radial_features=radial_features,
        contour_path=contour_path,
        shape_topology=shape_topology,
        shape_narrative=shape_narrative,
    )


def _generate_shape_topology(
    radial_features: str, skeleton_desc: str, ctx: PipelineContext
) -> str:
    """Synthesize radial profile peaks + skeleton into a shape topology summary.

    Describes the overall shape in terms of number, size, and direction
    of extensions from the core body mass.
    """
    import re

    if not radial_features or "No significant" in radial_features:
        return ""

    # Parse peaks from radial_features string
    # Format: "Peaks: 40° (32px, upper-right, broad lobe ~70° wide), ..."
    peak_pattern = re.compile(
        r"(\d+)\u00b0\s*\((\d+)px,\s*([^,]+),\s*([^~]+)~(\d+)\u00b0"
    )
    peaks = peak_pattern.findall(radial_features)
    if not peaks:
        return ""

    # Classify peaks
    extensions = []
    for angle_str, dist_str, direction, width_type, width_str in peaks:
        width_type = width_type.strip()
        direction = direction.strip()
        width = int(width_str)
        dist = int(dist_str)

        if "broad" in width_type:
            size = "major broad"
        elif "moderate" in width_type:
            size = "moderate"
        else:
            size = "narrow"

        extensions.append({
            "angle": int(angle_str),
            "dist": dist,
            "direction": direction,
            "size": size,
            "width": width,
        })

    if not extensions:
        return ""

    # Build topology description
    parts = []

    # Count by category
    broad = [e for e in extensions if "broad" in e["size"]]
    moderate = [e for e in extensions if "moderate" in e["size"]]
    narrow = [e for e in extensions if "narrow" in e["size"]]

    ext_parts = []
    if broad:
        dirs = ", ".join(e["direction"] for e in broad)
        ext_parts.append(
            f"{len(broad)} dominant broad extension{'s' if len(broad) > 1 else ''} "
            f"({dirs}, ~{broad[0]['width']}\u00b0 of perimeter)"
        )
    if moderate:
        dirs = ", ".join(e["direction"] for e in moderate)
        ext_parts.append(f"{len(moderate)} moderate extension{'s' if len(moderate) > 1 else ''} ({dirs})")
    if narrow:
        dirs = ", ".join(e["direction"] for e in narrow)
        ext_parts.append(f"{len(narrow)} narrow spike{'s' if len(narrow) > 1 else ''} ({dirs})")

    parts.append("body mass with " + ", ".join(ext_parts))

    # Add skeleton branch info if relevant
    branch_match = re.search(r"(\d+)\s*branches", skeleton_desc)
    junction_match = re.search(r"(\d+)\s*junctions", skeleton_desc)
    if branch_match:
        branches = int(branch_match.group(1))
        junctions = int(junction_match.group(1)) if junction_match else 0
        if branches >= _COMPLEX_BRANCH_THRESHOLD:
            parts.append(f"complex internal branching ({branches} branches, {junctions} junctions)")
        elif branches >= _MODERATE_BRANCH_THRESHOLD:
            parts.append(f"moderate branching ({branches} branches)")

    # Focal elements
    focal_count = 0
    for sp in ctx.subpaths:
        concentric = sp.features.get("concentric_groups", [])
        for g in concentric:
            center = g.get("center", (0, 0))
            members = g.get("members", [])
            if len(members) >= 2:
                cx, cy = center
                if (_CANVAS_EDGE_MARGIN_PCT * ctx.canvas_width < cx < (1 - _CANVAS_EDGE_MARGIN_PCT) * ctx.canvas_width
                        and _CANVAS_EDGE_MARGIN_PCT * ctx.canvas_height < cy < (1 - _CANVAS_EDGE_MARGIN_PCT) * ctx.canvas_height):
                    focal_count += 1
        break  # concentric_groups is on first subpath

    if focal_count > 0:
        parts.append(f"{focal_count} circular focal feature{'s' if focal_count > 1 else ''}")

    return "; ".join(parts)


def _generate_summary(
    ctx: PipelineContext,
    orientation: str,
    focal_elements: list[str],
    reading_hints: list[str],
) -> str:
    """Generate a concise layout summary synthesizing orientation + structure.

    Describes composition shape, dominant axis, structural layering, and mass
    distribution in neutral geometric terms.
    """
    import re

    parts = []

    # Orientation summary
    if "strong bilateral symmetry" in orientation.lower():
        parts.append("symmetric composition (strong bilateral)")
    elif "asymmetric" in orientation.lower():
        # Extract mass direction
        if "mass concentrated right" in orientation.lower():
            parts.append("asymmetric composition (mass toward right)")
        elif "mass concentrated left" in orientation.lower():
            parts.append("asymmetric composition (mass toward left)")
        else:
            parts.append("asymmetric composition")

    # Mass distribution from grid
    grid = _parse_grid(ctx.ascii_grid_positive)
    if grid:
        row_profile = _row_profile(grid)
        if row_profile and len(row_profile) >= 6:
            n_rows = len(row_profile)
            third = n_rows // 3
            top_fill = sum(row_profile[:third])
            mid_fill = sum(row_profile[third:2*third])
            bot_fill = sum(row_profile[2*third:])
            total = top_fill + mid_fill + bot_fill
            if total > 0:
                bot_pct = bot_fill / total
                mid_pct = mid_fill / total
                top_pct = top_fill / total
                if bot_pct > _MASS_CONCENTRATION_THRESHOLD:
                    parts.append("mass concentrated at bottom")
                elif top_pct > _MASS_CONCENTRATION_THRESHOLD:
                    parts.append("mass concentrated at top")
                elif mid_pct > _MASS_CONCENTRATION_THRESHOLD:
                    parts.append("mass concentrated at middle")

    # Count structural layers from reading hints
    layer_count = 0
    for hint in reading_hints:
        match = re.search(r"Major structural elements.*?:\s*(.*)", hint)
        if match:
            layer_count = match.group(1).count(";") + 1

    if layer_count >= _MIN_STRUCTURAL_LAYERS:
        parts.append(f"{layer_count} structural layers")

    # Background element info
    bg_area_pct = ""
    bg_extends_up = False
    bg_wide = False
    for hint in reading_hints:
        if "LARGER than primary boundary" in hint:
            match = re.search(r"area=(\d+)% of primary", hint)
            if match:
                bg_area_pct = match.group(1) + "%"
        if "Background element layout" in hint:
            if "extends up" in hint.lower() or "upper half" in hint.lower():
                bg_extends_up = True
            if "spans" in hint.lower() and "%" in hint:
                bg_wide = True

    bg_parts = []
    if bg_extends_up:
        bg_parts.append("extends into upper half")
    if bg_wide:
        bg_parts.append("spans wide")
    if bg_area_pct:
        bg_parts.append(f"area={bg_area_pct} of primary (LARGER than primary boundary)")

    if bg_parts:
        parts.append(f"large background element: {', '.join(bg_parts)}")

    # Focal element offset from center
    for fe in focal_elements:
        if "offset from center" in fe.lower():
            parts.append(fe.strip())
            break

    if not parts:
        return ""

    return "; ".join(parts)


# ── Grid parsing ──


def _parse_grid(grid_text: str) -> list[list[bool]]:
    """Parse ASCII grid into 2D boolean array. X=True, .=False."""
    if not grid_text:
        return []
    rows = []
    for line in grid_text.strip().split("\n"):
        row = []
        for ch in line:
            if ch == "X":
                row.append(True)
            elif ch == ".":
                row.append(False)
            # skip spaces
        rows.append(row)
    return rows


def _grid_stats(grid: list[list[bool]]) -> dict:
    """Compute basic grid statistics."""
    if not grid:
        return {"rows": 0, "cols": 0, "fill_pct": 0.0}

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    total = 0
    filled = 0

    for row in grid:
        for cell in row:
            total += 1
            if cell:
                filled += 1

    return {
        "rows": rows,
        "cols": cols,
        "total": total,
        "filled": filled,
        "fill_pct": (filled / total * 100) if total > 0 else 0.0,
    }


def _quadrant_fill(grid: list[list[bool]]) -> dict[str, float]:
    """Compute fill percentage per quadrant (UL, UR, LL, LR)."""
    if not grid:
        return {"UL": 0, "UR": 0, "LL": 0, "LR": 0}

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    mid_r, mid_c = rows // 2, cols // 2

    quads = {"UL": [0, 0], "UR": [0, 0], "LL": [0, 0], "LR": [0, 0]}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if r < mid_r:
                q = "UL" if c < mid_c else "UR"
            else:
                q = "LL" if c < mid_c else "LR"
            quads[q][1] += 1
            if cell:
                quads[q][0] += 1

    return {
        k: (v[0] / v[1] * 100 if v[1] > 0 else 0.0)
        for k, v in quads.items()
    }


def _column_profile(grid: list[list[bool]]) -> list[float]:
    """Fill percentage per column (left-to-right mass profile)."""
    if not grid:
        return []
    cols = max(len(r) for r in grid)
    profile = []
    for c in range(cols):
        filled = sum(1 for row in grid if c < len(row) and row[c])
        profile.append(filled / len(grid) * 100)
    return profile


def _row_profile(grid: list[list[bool]]) -> list[float]:
    """Fill percentage per row (top-to-bottom mass profile)."""
    if not grid:
        return []
    cols = max(len(r) for r in grid) if grid else 1
    profile = []
    for row in grid:
        filled = sum(1 for cell in row if cell)
        profile.append(filled / cols * 100)
    return profile


def _find_protrusions(grid: list[list[bool]]) -> list[dict]:
    """Find regions that protrude from the main mass."""
    if not grid:
        return []

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    col_profile = _column_profile(grid)
    row_profile = _row_profile(grid)

    # Median fill gives us the "core" density
    col_median = float(np.median(col_profile)) if col_profile else 50.0
    row_median = float(np.median(row_profile)) if row_profile else 50.0

    protrusions = []

    # Check top edge — find columns with fill in top rows but not in main body
    top_extent = 0
    for r in range(rows):
        if row_profile[r] > _PROFILE_FILL_PCT:
            top_extent = r
            break

    if top_extent > _PROTRUSION_MIN_EXTENT:
        # Top protrusion: find which columns have fill in the top region
        top_cols_filled = []
        for c in range(cols):
            top_fill = sum(1 for r in range(top_extent) if r < rows and c < len(grid[r]) and grid[r][c])
            if top_fill > 0:
                top_cols_filled.append(c)
        if top_cols_filled:
            region = "left" if np.mean(top_cols_filled) < cols / 2 else "right"
            if abs(np.mean(top_cols_filled) - cols / 2) < cols * _CENTER_TOLERANCE_PCT:
                region = "center"
            protrusions.append({
                "direction": "upward",
                "region": region,
                "extent": top_extent,
                "width_cols": len(top_cols_filled),
            })

    # Check bottom edge
    bottom_extent = 0
    for r in range(rows - 1, -1, -1):
        if row_profile[r] > _PROFILE_FILL_PCT:
            bottom_extent = rows - 1 - r
            break

    if bottom_extent > _PROTRUSION_MIN_EXTENT:
        bottom_cols_filled = []
        for c in range(cols):
            bottom_fill = sum(
                1 for r in range(rows - bottom_extent, rows)
                if c < len(grid[r]) and grid[r][c]
            )
            if bottom_fill > 0:
                bottom_cols_filled.append(c)
        if bottom_cols_filled:
            region = "left" if np.mean(bottom_cols_filled) < cols / 2 else "right"
            if abs(np.mean(bottom_cols_filled) - cols / 2) < cols * _CENTER_TOLERANCE_PCT:
                region = "center"
            protrusions.append({
                "direction": "downward",
                "region": region,
                "extent": bottom_extent,
            })

    # Check left edge
    left_sparse = sum(1 for c in range(min(_EDGE_SAMPLE_COLS, cols)) if col_profile[c] < _SPARSE_COL_FILL_PCT)
    right_sparse = sum(1 for c in range(max(0, cols - _EDGE_SAMPLE_COLS), cols) if col_profile[c] < _SPARSE_COL_FILL_PCT)

    if left_sparse > _SPARSE_EDGE_MIN_COUNT:
        protrusions.append({"direction": "left_sparse", "note": "left edge has low fill"})
    if right_sparse > _SPARSE_EDGE_MIN_COUNT:
        protrusions.append({"direction": "right_sparse", "note": "right edge has low fill"})

    return protrusions


def _compute_principal_axis(grid: list[list[bool]]) -> dict:
    """Compute principal axis of the filled mass using 2nd-moment (covariance).

    Returns angle in degrees (0°=horizontal, 90°=vertical) and eccentricity.
    Eccentricity near 1.0 = strongly oriented; near 0.0 = roughly circular mass.
    """
    if not grid:
        return {"angle": 0.0, "eccentricity": 0.0, "orientation": "unknown"}

    # Collect filled cell coordinates
    coords = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell:
                coords.append((c, r))  # (x, y)

    if len(coords) < 3:
        return {"angle": 0.0, "eccentricity": 0.0, "orientation": "unknown"}

    arr = np.array(coords, dtype=float)
    # Center of mass
    cx = np.mean(arr[:, 0])
    cy = np.mean(arr[:, 1])

    # Central moments (covariance matrix)
    dx = arr[:, 0] - cx
    dy = arr[:, 1] - cy
    mu20 = np.mean(dx * dx)
    mu02 = np.mean(dy * dy)
    mu11 = np.mean(dx * dy)

    # Eigenvalues of covariance matrix → principal axes
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11 * mu11
    discriminant = max(0, trace * trace / 4 - det)
    sqrt_disc = discriminant ** 0.5
    lambda1 = trace / 2 + sqrt_disc  # larger eigenvalue
    lambda2 = trace / 2 - sqrt_disc  # smaller eigenvalue

    # Angle of principal axis (dominant direction of spread)
    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = float(np.degrees(angle_rad))

    # Eccentricity: how elongated the mass distribution is
    if lambda1 > 0:
        eccentricity = float(1.0 - lambda2 / lambda1)
    else:
        eccentricity = 0.0

    # Classify orientation
    # angle_deg is in [-90, 90]. Near 0 = horizontal spread, near ±90 = vertical spread.
    abs_angle = abs(angle_deg)
    if abs_angle > _ORIENT_VERTICAL_MIN_DEG:
        orientation = "vertical"
    elif abs_angle < _ORIENT_HORIZONTAL_MAX_DEG:
        orientation = "horizontal"
    else:
        orientation = "diagonal"

    return {
        "angle": angle_deg,
        "eccentricity": eccentricity,
        "orientation": orientation,
    }


def _compute_boundary_lobes(grid: list[list[bool]]) -> dict:
    """Compute the centroid distance function and count boundary lobes.

    Samples the distance from the centroid to the boundary at regular angles.
    Local maxima in this function correspond to "lobes" — distinct protruding
    mass regions (head, tail, limbs, etc.). Multi-lobed = complex silhouette.
    """
    if not grid:
        return {"lobe_count": 0, "max_min_ratio": 1.0}

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    if rows < _MIN_GRID_DIM or cols < _MIN_GRID_DIM:
        return {"lobe_count": 0, "max_min_ratio": 1.0}

    # Find centroid of filled region
    filled_coords = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell:
                filled_coords.append((c, r))

    if len(filled_coords) < _MIN_FILLED_FOR_CENTROID:
        return {"lobe_count": 0, "max_min_ratio": 1.0}

    arr = np.array(filled_coords, dtype=float)
    cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))

    # Sample distance from centroid to boundary at N angles
    n_angles = _LOBE_ANGULAR_SAMPLES
    distances = []
    for i in range(n_angles):
        angle = 2 * np.pi * i / n_angles
        dx, dy = np.cos(angle), np.sin(angle)
        # Walk outward from centroid until hitting empty space or boundary
        max_dist = 0
        for step in range(1, max(rows, cols)):
            px = int(cx + dx * step)
            py = int(cy + dy * step)
            if px < 0 or px >= cols or py < 0 or py >= rows:
                max_dist = step
                break
            if py < len(grid) and px < len(grid[py]) and grid[py][px]:
                max_dist = step
            else:
                break
        distances.append(max_dist)

    if not distances or max(distances) == 0:
        return {"lobe_count": 0, "max_min_ratio": 1.0}

    # Smooth to reduce noise
    kernel = np.ones(_SMOOTHING_KERNEL_TAPS) / _SMOOTHING_KERNEL_TAPS
    # Circular smoothing: pad the signal
    padded = distances + distances + distances
    smoothed_full = np.convolve(padded, kernel, mode='same')
    smoothed = smoothed_full[n_angles:2 * n_angles]

    # Count local maxima (lobes) — peaks above Weber's JND fraction of range
    min_dist = min(smoothed)
    max_dist = max(smoothed)
    threshold = min_dist + (max_dist - min_dist) * _LOBE_PROMINENCE_FRACTION

    lobes = 0
    in_peak = False
    for i in range(n_angles):
        if smoothed[i] > threshold and not in_peak:
            in_peak = True
            lobes += 1
        elif smoothed[i] <= threshold:
            in_peak = False

    ratio = float(max_dist / min_dist) if min_dist > 0 else float('inf')

    return {"lobe_count": lobes, "max_min_ratio": ratio}


def _compute_convex_hull_deficiency(grid: list[list[bool]]) -> float:
    """Ratio of filled area to convex hull area. Low = many protrusions/concavities."""
    if not grid:
        return 1.0

    coords = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell:
                coords.append((c, r))

    if len(coords) < 3:
        return 1.0

    from scipy.spatial import ConvexHull

    try:
        arr = np.array(coords, dtype=float)
        hull = ConvexHull(arr)
        hull_area = hull.volume  # In 2D, ConvexHull.volume = area
        filled_area = len(coords)
        return float(filled_area / hull_area) if hull_area > 0 else 1.0
    except Exception:
        return 1.0


# ── Interpretation functions ──


def _interpret_silhouette(ctx: PipelineContext, grid: list[list[bool]]) -> str:
    """Describe the overall silhouette shape from the positive space grid."""
    if not grid:
        return "no grid data available"

    stats = _grid_stats(grid)
    quads = _quadrant_fill(grid)
    col_profile = _column_profile(grid)
    row_profile = _row_profile(grid)

    # Overall shape
    fill = stats["fill_pct"]
    parts = []

    if fill > _FILL_DENSE_PCT:
        parts.append("dense, nearly rectangular fill")
    elif fill > _FILL_MOSTLY_PCT:
        parts.append("mostly filled with shaped contour")
    elif fill > _FILL_MODERATE_PCT:
        parts.append("moderate fill with clear negative space")
    else:
        parts.append("sparse, open composition")

    # Widest/narrowest points — skip for very dense fills where widest is meaningless
    if col_profile and fill < _FILL_WIDEST_CUTOFF:
        widest_row = int(np.argmax(row_profile))
        narrowest_filled = [i for i, v in enumerate(row_profile) if v > _FILL_OCCUPIED_MIN_PCT]
        if narrowest_filled:
            row_count = len(row_profile)
            widest_pos = "top" if widest_row < row_count * _THIRD else ("middle" if widest_row < row_count * _TWO_THIRDS else "bottom")
            parts.append(f"widest at {widest_pos}")
    elif col_profile and fill >= _FILL_WIDEST_CUTOFF:
        # Dense fill — describe where the silhouette NARROWS or has gaps instead
        row_count = len(row_profile)
        # Find which rows have the least fill (that's where the silhouette shape is)
        bottom_third_fill = np.mean(row_profile[2 * row_count // 3:]) if row_count > 3 else 100
        top_third_fill = np.mean(row_profile[:row_count // 3]) if row_count > 3 else 100
        if bottom_third_fill < top_third_fill - _FILL_NARROWING_DIFF:
            parts.append("narrows toward bottom")
        elif top_third_fill < bottom_third_fill - _FILL_NARROWING_DIFF:
            parts.append("narrows toward top")

    # Vertical extent vs horizontal
    filled_rows = [i for i, v in enumerate(row_profile) if v > _FILL_OCCUPIED_MIN_PCT]
    filled_cols = [i for i, v in enumerate(col_profile) if v > _FILL_OCCUPIED_MIN_PCT]
    if filled_rows and filled_cols:
        v_span = filled_rows[-1] - filled_rows[0] + 1
        h_span = filled_cols[-1] - filled_cols[0] + 1
        if v_span > h_span * _ASPECT_THRESHOLD:
            parts.append("taller than wide")
        elif h_span > v_span * _ASPECT_THRESHOLD:
            parts.append("wider than tall")
        else:
            parts.append("roughly square proportions")

    # Asymmetry
    left_fill = sum(col_profile[:len(col_profile) // 2])
    right_fill = sum(col_profile[len(col_profile) // 2:])
    total_fill = left_fill + right_fill
    if total_fill > 0:
        balance = left_fill / total_fill
        if balance > _BALANCE_HEAVY_THRESHOLD:
            parts.append("heavier on left side")
        elif balance < _BALANCE_LIGHT_THRESHOLD:
            parts.append("heavier on right side")

    # Protrusions
    protrusions = _find_protrusions(grid)
    for p in protrusions:
        if p["direction"] == "upward":
            parts.append(f"upward protrusion from {p['region']}")
        elif p["direction"] == "downward":
            parts.append(f"extends down on {p['region']}")

    # Width profile: find narrowest point ("waist") and distinct mass regions
    if row_profile and len(row_profile) >= _MIN_ROWS_WAIST:
        # Smooth the profile to avoid noise
        kernel_size = max(3, len(row_profile) // _MIN_ROWS_WAIST)
        smoothed = np.convolve(row_profile, np.ones(kernel_size) / kernel_size, mode='same')
        # Find local minima (valleys) in the smoothed profile where fill > barely visible
        # Only consider rows that are within the filled region
        filled_range = [i for i, v in enumerate(smoothed) if v > _GAP_REPORT_MIN_PCT]
        if len(filled_range) >= _MIN_WAIST_RANGE:
            start_r, end_r = filled_range[0], filled_range[-1]
            inner = smoothed[start_r:end_r + 1]
            if len(inner) >= _MIN_WAIST_RANGE:
                # Find the deepest valley (narrowest point between two wider regions)
                min_val = float('inf')
                min_pos = 0
                # Only look between 20% and 80% of the filled region
                search_start = len(inner) // 5
                search_end = 4 * len(inner) // 5
                for i in range(search_start, search_end):
                    if inner[i] < min_val:
                        min_val = inner[i]
                        min_pos = i
                # Check if there's a real "waist" — the narrowest point must be notably narrower
                max_above = max(inner[:min_pos]) if min_pos > 0 else 0
                max_below = max(inner[min_pos:]) if min_pos < len(inner) else 0
                wider = min(max_above, max_below)
                if wider > 0 and min_val < wider * _WAIST_NARROW_RATIO:
                    waist_row = start_r + min_pos
                    waist_pct = waist_row / len(row_profile) * 100
                    parts.append(
                        f"narrowest at row {waist_pct:.0f}% (width drops to {min_val:.0f}% vs {wider:.0f}% above/below) — distinct upper and lower mass regions"
                    )

    # Empty corners/edges — tells the LLM where the figure DOESN'T extend
    quads = _quadrant_fill(grid)
    quad_names = {"UL": "upper-left", "UR": "upper-right", "LL": "lower-left", "LR": "lower-right"}
    empty_quads = [quad_names[k] for k, v in quads.items() if v < _QUAD_EMPTY_PCT]
    sparse_quads = [quad_names[k] for k, v in quads.items() if _QUAD_EMPTY_PCT <= v < _QUAD_SPARSE_PCT]
    if empty_quads:
        parts.append(f"empty: {', '.join(empty_quads)}")
    if sparse_quads:
        parts.append(f"sparse: {', '.join(sparse_quads)}")

    # Bottom/top edge readings — which side of the bottom has fill
    # This helps distinguish sitting figures, grounded objects, etc.
    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    if rows > _MIN_GRID_DIM and cols > _MIN_GRID_DIM:
        bottom_rows = grid[max(0, rows - _EDGE_SAMPLE_COLS):]
        left_bottom = sum(
            1 for r in bottom_rows for c in range(min(cols // 2, len(r))) if c < len(r) and r[c]
        )
        right_bottom = sum(
            1 for r in bottom_rows for c in range(cols // 2, min(cols, len(r))) if c < len(r) and r[c]
        )
        total_bottom = left_bottom + right_bottom
        if total_bottom > 0:
            left_pct = left_bottom / total_bottom
            if left_pct < _BOTTOM_EDGE_LOW:
                parts.append("bottom edge: right side only (left side open)")
            elif left_pct > _BOTTOM_EDGE_HIGH:
                parts.append("bottom edge: left side only (right side open)")
            elif left_pct < _COM_LOW_PCT:
                parts.append("bottom edge: mostly right")
            elif left_pct > _COM_HIGH_PCT:
                parts.append("bottom edge: mostly left")

    # Principal axis of mass — tells you the dominant orientation of the shape
    paxis = _compute_principal_axis(grid)
    if paxis["orientation"] != "unknown":
        parts.append(
            f"principal axis: {paxis['orientation']} ({paxis['angle']:.0f}°, "
            f"eccentricity={paxis['eccentricity']:.2f})"
        )

    # Convex hull deficiency — how much the shape deviates from its convex hull
    hull_def = _compute_convex_hull_deficiency(grid)
    if hull_def < _HULL_HIGHLY_CONCAVE:
        parts.append(f"highly concave silhouette (hull fill={hull_def:.0%})")
    elif hull_def < _HULL_MODERATELY_CONCAVE:
        parts.append(f"moderately concave (hull fill={hull_def:.0%})")

    # Boundary lobe analysis — how many distinct mass regions protrude from center
    lobes = _compute_boundary_lobes(grid)
    if lobes["lobe_count"] >= 2:
        parts.append(
            f"boundary lobes: {lobes['lobe_count']} distinct protruding regions "
            f"(max/min distance ratio={lobes['max_min_ratio']:.1f})"
        )

    return "; ".join(parts)


def _interpret_orientation(ctx: PipelineContext, grid: list[list[bool]]) -> str:
    """Determine orientation from symmetry and mass distribution."""
    sym_score = ctx.symmetry_score
    sym_axis = ctx.symmetry_axis or "none"

    # Mass balance from grid
    col_profile = _column_profile(grid)
    if col_profile:
        mid = len(col_profile) // 2
        left_mass = sum(col_profile[:mid])
        right_mass = sum(col_profile[mid:])
        total = left_mass + right_mass
        balance = left_mass / total if total > 0 else 0.5
    else:
        balance = 0.5

    parts = []

    # Symmetry interpretation
    if sym_score >= _SYM_STRONG:
        parts.append("strong bilateral symmetry")
        parts.append(f"strong {sym_axis} symmetry ({sym_score:.2f})")
    elif sym_score >= _SYM_MODERATE:
        parts.append("semi-symmetric")
        if abs(balance - 0.5) > _BALANCE_MINOR_OFFSET:
            side = "left" if balance > _BALANCE_SIDE_THRESHOLD else "right"
            parts.append(f"mass concentrated {side}")
        else:
            parts.append("roughly centered but with asymmetric features")
        parts.append(f"moderate {sym_axis} symmetry ({sym_score:.2f})")
    elif sym_score >= _SYM_WEAK:
        if abs(balance - 0.5) > _CENTER_TOLERANCE_PCT:
            side = "left" if balance > _BALANCE_SIDE_THRESHOLD else "right"
            parts.append(f"asymmetric, mass concentrated {side}")
        else:
            parts.append("asymmetric composition")
        parts.append(f"weak symmetry ({sym_score:.2f})")
    else:
        parts.append("highly asymmetric / abstract")
        parts.append(f"minimal symmetry ({sym_score:.2f})")

    return "; ".join(parts)


def _interpret_composition(ctx: PipelineContext) -> str:
    """Classify the composition type based on element count and structure."""
    n = len(ctx.subpaths)

    # Nesting depth from containment
    max_depth = 0
    if ctx.containment_matrix is not None:
        for i in range(n):
            depth = sum(1 for j in range(n) if j != i and ctx.containment_matrix[j][i])
            max_depth = max(max_depth, depth)

    # Shape variety
    shape_classes = set()
    for sp in ctx.subpaths:
        shape_classes.add(sp.features.get("shape_class", "organic"))

    # Classify
    if n <= _COMP_SIMPLE_MAX:
        comp = "simple icon"
    elif n <= _COMP_STRUCTURED_MAX:
        comp = "structured icon or logo"
    elif n <= _COMP_DETAILED_MAX:
        comp = "detailed icon or illustration"
    else:
        comp = "complex illustration"

    if max_depth >= _NESTING_DEEP:
        comp += f" with deep nesting ({max_depth} levels)"
    elif max_depth >= _NESTING_MODERATE:
        comp += " with nested elements"

    if len(shape_classes) >= _MIXED_SHAPES_MIN:
        comp += f", mixed shapes ({', '.join(sorted(shape_classes))})"
    elif len(shape_classes) == 1:
        comp += f", uniform {list(shape_classes)[0]} shapes"

    return comp


def _analyze_mass_distribution(grid: list[list[bool]]) -> str:
    """Describe where visual weight is concentrated."""
    if not grid:
        return "unknown"

    quads = _quadrant_fill(grid)
    col_profile = _column_profile(grid)
    row_profile = _row_profile(grid)

    parts = []

    # Find dominant quadrant(s)
    max_fill = max(quads.values())
    dominant = [k for k, v in quads.items() if v > max_fill * _DOMINANT_QUAD_RATIO]

    quad_names = {"UL": "upper-left", "UR": "upper-right", "LL": "lower-left", "LR": "lower-right"}
    if len(dominant) == 4:
        parts.append("evenly distributed")
    elif len(dominant) <= 2:
        parts.append(f"concentrated in {', '.join(quad_names[d] for d in dominant)}")
    else:
        parts.append("broadly distributed")

    # Vertical center of mass
    if row_profile:
        total_mass = sum(row_profile)
        if total_mass > 0:
            com_row = sum(i * v for i, v in enumerate(row_profile)) / total_mass
            row_pos = com_row / len(row_profile)
            if row_pos < _COM_LOW_PCT:
                parts.append("vertically top-heavy")
            elif row_pos > _COM_HIGH_PCT:
                parts.append("vertically bottom-heavy")
            else:
                parts.append("vertically centered")

    # Horizontal center of mass
    if col_profile:
        total_mass = sum(col_profile)
        if total_mass > 0:
            com_col = sum(i * v for i, v in enumerate(col_profile)) / total_mass
            col_pos = com_col / len(col_profile)
            if col_pos < _COM_LOW_PCT:
                parts.append("horizontally left-leaning")
            elif col_pos > _COM_HIGH_PCT:
                parts.append("horizontally right-leaning")

    # Fill percentage per quadrant
    quad_strs = [f"{quad_names[k]}={v:.0f}%" for k, v in quads.items()]
    parts.append(f"quadrants: {', '.join(quad_strs)}")

    # Vertical thirds — where is the mass concentrated?
    if row_profile and len(row_profile) >= 6:
        n_rows = len(row_profile)
        third = n_rows // 3
        top_fill = sum(row_profile[:third])
        mid_fill = sum(row_profile[third:2*third])
        bot_fill = sum(row_profile[2*third:])
        total = top_fill + mid_fill + bot_fill
        if total > 0:
            top_pct = top_fill / total * 100
            mid_pct = mid_fill / total * 100
            bot_pct = bot_fill / total * 100
            parts.append(f"vertical thirds: top={top_pct:.0f}%, mid={mid_pct:.0f}%, bottom={bot_pct:.0f}%")

    return "; ".join(parts)


def _identify_focal_elements(ctx: PipelineContext) -> list[str]:
    """Find elements that likely represent key visual features."""
    focal = []

    canvas_w, canvas_h = ctx.canvas_width, ctx.canvas_height

    # Concentric groups — multi-ring constructions (most structurally significant)
    concentric = ctx.subpaths[0].features.get("concentric_groups", []) if ctx.subpaths else []
    concentric_entries: list[tuple[float, tuple[float, float], str]] = []  # (circ, center, text)
    for group in concentric:
        members = group.get("members", [])
        center = group.get("center", (0, 0))
        if len(members) >= 2:
            member_sps = [sp for sp in ctx.subpaths if sp.id in members]
            circs = [sp.features.get("circularity", 0) for sp in member_sps]
            max_circ = max(circs) if circs else 0
            if max_circ > _CIRC_MODERATE:
                tiers = [sp.features.get("size_tier", "MEDIUM") for sp in member_sps]
                # Describe position on canvas
                pos_x = "left" if center[0] < canvas_w * _THIRD else ("center" if center[0] < canvas_w * _TWO_THIRDS else "right")
                pos_y = "top" if center[1] < canvas_h * _THIRD else ("mid" if center[1] < canvas_h * _TWO_THIRDS else "bottom")

                # Containment context — which structural element contains this group?
                parent_info = ""
                if ctx.containment_matrix is not None:
                    first_id = members[0]
                    first_idx = next((i for i, s in enumerate(ctx.subpaths) if s.id == first_id), -1)
                    if first_idx >= 0:
                        member_set = set(members)
                        parents = [
                            i for i in range(len(ctx.subpaths))
                            if i != first_idx
                            and ctx.subpaths[i].id not in member_set
                            and ctx.containment_matrix[i][first_idx]
                        ]
                        if parents:
                            parent_idx = min(parents, key=lambda i: ctx.subpaths[i].area)
                            parent_info = f", inside {ctx.subpaths[parent_idx].id}"

                # Edge annotation — groups at canvas edge are likely decorative
                _edge_px = _CANVAS_EDGE_MARGIN_PCT * min(canvas_w, canvas_h)
                at_edge = (center[0] < _edge_px or center[0] > canvas_w - _edge_px or
                           center[1] < _edge_px or center[1] > canvas_h - _edge_px)
                if at_edge:
                    edge_info = ", at canvas edge"
                elif not parent_info:
                    edge_info = ", root-level (independent, drawn on top)"
                else:
                    edge_info = ""

                entry = (
                    f"Concentric group [{', '.join(members)}] at ({center[0]:.0f},{center[1]:.0f}) [{pos_y}-{pos_x}], "
                    f"{len(members)} layers, max circularity={max_circ:.2f}, tiers={'/'.join(tiers)}{parent_info}{edge_info}"
                )
                concentric_entries.append((max_circ, center, entry))
    # Show top 4 by circularity
    concentric_entries.sort(key=lambda x: x[0], reverse=True)
    for _, _, entry in concentric_entries[:4]:
        focal.append(entry)

    # Add spatial relationships from the most circular group to others
    if len(concentric_entries) >= 2:
        # Only show relationships from the top group (highest circularity) to others
        # Skip groups near canvas edges (likely decorative, not structural)
        top = concentric_entries[0]
        relations = []
        for entry in concentric_entries[1:4]:
            c1 = top[1]
            c2 = entry[1]
            # Skip if either group is at canvas edge
            edge_margin = min(canvas_w, canvas_h) * _CANVAS_EDGE_MARGIN_PCT
            if (c2[0] < edge_margin or c2[0] > canvas_w - edge_margin) and \
               (c2[1] < edge_margin or c2[1] > canvas_h - edge_margin):
                continue
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            if abs(dy) < canvas_h * _DIRECTION_TOLERANCE_PCT:
                direction = "right of" if dx > 0 else "left of"
            elif abs(dx) < canvas_w * _DIRECTION_TOLERANCE_PCT:
                direction = "below" if dy > 0 else "above"
            else:
                v = "below" if dy > 0 else "above"
                h = "right" if dx > 0 else "left"
                direction = f"{v}-{h} of"
            dist = (dx**2 + dy**2) ** 0.5
            relations.append(
                f"  -> group at ({c2[0]:.0f},{c2[1]:.0f}) is {direction} group at ({c1[0]:.0f},{c1[1]:.0f}), {dist:.0f}px apart"
            )
        focal.extend(relations[:3])

    # Individual high-circularity elements (not already covered by concentric groups)
    concentric_ids = set()
    for group in concentric:
        concentric_ids.update(group.get("members", []))

    for sp in ctx.subpaths:
        if sp.id in concentric_ids:
            continue
        circ = sp.features.get("circularity", 0.0)
        tier = sp.features.get("size_tier", "MEDIUM")
        shape = sp.features.get("shape_class", "organic")
        cx, cy = sp.centroid

        if circ > _CIRC_HIGH and tier in ("SMALL", "MEDIUM"):
            contained_by = sp.features.get("contained_by", [])
            if contained_by:
                focal.append(
                    f"{sp.id}: circular ({shape}, circ={circ:.2f}) inside {contained_by[0]}"
                )
            else:
                focal.append(
                    f"{sp.id}: circular ({shape}, circ={circ:.2f}) at ({cx:.0f},{cy:.0f}), standalone"
                )

    # Offset indicator: if the highest-circularity root-level concentric group
    # is far from canvas center, note the offset.
    if concentric_entries and canvas_w > 0:
        _edge_px2 = _CANVAS_EDGE_MARGIN_PCT * min(canvas_w, canvas_h)
        for circ_val, center, entry_text in concentric_entries:
            at_edge = (center[0] < _edge_px2 or center[0] > canvas_w - _edge_px2 or
                       center[1] < _edge_px2 or center[1] > canvas_h - _edge_px2)
            if at_edge:
                continue
            if "root-level" in entry_text or "inside" not in entry_text:
                x_pct = center[0] / canvas_w
                if x_pct > _POS_THREE_QUARTER or x_pct < _POS_QUARTER:
                    side = "right" if x_pct > 0.5 else "left"
                    focal.append(
                        f"Highest-circularity concentric group at x={x_pct:.0%} — "
                        f"offset from center (far {side})"
                    )
                break

    # Limit to most interesting
    return focal[:_TOP_FEATURES_N + 2]


def _detect_protrusions_from_elements(ctx: PipelineContext, grid: list[list[bool]]) -> list[str]:
    """Detect elements that protrude from the core mass."""
    results = []

    if not ctx.subpaths or not grid:
        return results

    canvas_w, canvas_h = ctx.canvas_width, ctx.canvas_height

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    if rows == 0 or cols == 0:
        return results

    row_fills = _row_profile(grid)
    col_fills = _column_profile(grid)

    core_rows = [i for i, v in enumerate(row_fills) if v > _QUAD_EMPTY_PCT]
    core_cols = [i for i, v in enumerate(col_fills) if v > _QUAD_EMPTY_PCT]

    if not core_rows or not core_cols:
        return results

    core_top = core_rows[0] / rows * canvas_h
    core_bottom = core_rows[-1] / rows * canvas_h
    core_left = core_cols[0] / cols * canvas_w
    core_right = core_cols[-1] / cols * canvas_w
    margin_x = canvas_w * _PROTRUSION_MARGIN_PCT
    margin_y = canvas_h * _PROTRUSION_MARGIN_PCT

    for sp in ctx.subpaths:
        cx, cy = sp.centroid
        tier = sp.features.get("size_tier", "MEDIUM")
        ar = sp.features.get("aspect_ratio", 1.0)

        direction = ""
        if cy < core_top - margin_y:
            direction = "above"
        elif cy > core_bottom + margin_y:
            direction = "below"
        elif cx < core_left - margin_x:
            direction = "left of"
        elif cx > core_right + margin_x:
            direction = "right of"
        else:
            continue

        # Classify shape type using existing geometric features
        circ = sp.features.get("circularity", 0.0)
        curv_mean = sp.features.get("curvature_mean", 0.0)
        curv_max = sp.features.get("curvature_max", 0.0)
        corners = sp.features.get("corner_count", 0)
        inflections = sp.features.get("inflection_count", 0)
        eccentricity = ar if ar > 1 else (1 / ar if ar > 0 else 1)

        shape_type = _classify_shape_type(eccentricity, circ, curv_mean, curv_max, corners, inflections)
        pos = _pos_label(cx, cy, canvas_w, canvas_h)
        desc = f"{sp.id}: [{tier}] {shape_type} {direction} core mass at {pos}"
        if eccentricity > _ECCENTRIC_MODERATE:
            desc += f", elongation={eccentricity:.1f}"
        results.append(desc)

    return results[:_TOP_FEATURES_N]


def _generate_reading_hints(ctx: PipelineContext, grid: list[list[bool]]) -> list[str]:
    """Generate plain-language hints for interpreting the SVG."""
    hints = []
    n = len(ctx.subpaths)

    # Symmetry hint
    sym_score = ctx.symmetry_score
    if sym_score < _SYM_MODERATE:
        hints.append(
            f"Low symmetry ({sym_score:.2f}) — asymmetric composition, side-view, "
            "or angled arrangement."
        )
    elif sym_score > _SYM_READING_HIGH:
        hints.append(
            f"High symmetry ({sym_score:.2f}) — balanced/symmetric design or "
            "front-facing orientation."
        )

    # Mass distribution hint
    quads = _quadrant_fill(grid)
    if quads:
        imbalance = max(quads.values()) - min(quads.values())
        if imbalance > _QUAD_IMBALANCE_THRESHOLD:
            heavy = max(quads, key=quads.get)
            quad_names = {"UL": "upper-left", "UR": "upper-right", "LL": "lower-left", "LR": "lower-right"}
            hints.append(
                f"Visual weight is concentrated {quad_names[heavy]} "
                f"({quads[heavy]:.0f}% fill vs {min(quads.values()):.0f}% in lightest quadrant)."
            )

    # Containment depth
    if ctx.containment_matrix is not None:
        max_depth = 0
        for i in range(n):
            depth = sum(1 for j in range(n) if j != i and ctx.containment_matrix[j][i])
            max_depth = max(max_depth, depth)
        if max_depth >= _NESTING_DEEP:
            hints.append(f"Deep nesting ({max_depth} levels).")

    # Structural hierarchy — major top-level elements from stacking tree
    if ctx.subpaths and n > _DETAIL_ELEMENT_THRESHOLD:
        tree_text = ctx.subpaths[0].features.get("stacking_tree_text", "")
        if tree_text:
            root_ids = []
            for line in tree_text.strip().split("\n"):
                if line and not line[0].isspace():
                    eid = line.split(" ")[0].strip()
                    if eid.startswith("E"):
                        root_ids.append(eid)

            large_roots = []
            for rid in root_ids:
                sp = next((s for s in ctx.subpaths if s.id == rid), None)
                if sp and sp.features.get("size_tier") == "LARGE":
                    x1, y1, x2, y2 = sp.bbox
                    cov_w = (x2 - x1) / ctx.canvas_width * 100 if ctx.canvas_width > 0 else 0
                    cov_h = (y2 - y1) / ctx.canvas_height * 100 if ctx.canvas_height > 0 else 0

                    if cov_h > _COVERAGE_FULL and cov_w > _COVERAGE_FULL:
                        extent = "spans full canvas"
                    elif cov_h > _COVERAGE_MOST and cov_w > _COVERAGE_MOST:
                        extent = f"covers most of canvas ({cov_w:.0f}%w x {cov_h:.0f}%h)"
                    else:
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        h_pos = "left" if cx < ctx.canvas_width * _THIRD else ("center" if cx < ctx.canvas_width * _TWO_THIRDS else "right")
                        v_pos = "top" if cy < ctx.canvas_height * _THIRD else ("mid" if cy < ctx.canvas_height * _TWO_THIRDS else "bottom")
                        extent = f"{v_pos}-{h_pos}, {cov_w:.0f}%w x {cov_h:.0f}%h"

                    child_count = 0
                    z_index = -1
                    if ctx.containment_matrix is not None:
                        idx = next((i for i, s in enumerate(ctx.subpaths) if s.id == rid), -1)
                        if idx >= 0:
                            z_index = idx
                            child_count = sum(
                                1 for j in range(n)
                                if j != idx and ctx.containment_matrix[idx][j]
                            )

                    # Shape descriptor with precise measurements
                    shape_desc = ""
                    if sp:
                        conv = sp.features.get("convexity", 0)
                        asp = sp.features.get("aspect_ratio", 1.0)
                        sc = sp.features.get("shape_class", "organic")
                        if sc == "circular":
                            shape_desc = f"circular (conv={conv:.2f}, aspect={asp:.1f})"
                        elif conv > _CONVEX_COMPACT:
                            shape_desc = f"compact rounded (conv={conv:.2f}, aspect={asp:.1f})"
                        elif conv > _CONVEX_MODERATE:
                            if asp < _ASPECT_ELONGATED:
                                shape_desc = f"curved organic (conv={conv:.2f}, aspect={asp:.1f})"
                            else:
                                shape_desc = f"elongated (conv={conv:.2f}, aspect={asp:.1f})"
                        elif conv > _CONVEX_LOW:
                            if asp > _ASPECT_WIDE:
                                shape_desc = f"irregular spread (conv={conv:.2f}, aspect={asp:.1f})"
                            else:
                                shape_desc = f"irregular (conv={conv:.2f}, aspect={asp:.1f})"
                        else:
                            shape_desc = f"thin/branching (conv={conv:.2f}, aspect={asp:.1f})"

                    large_roots.append((rid, extent, child_count, z_index, shape_desc))

            if len(large_roots) >= 2:
                # Sort by z-index so draw order is clear
                large_roots.sort(key=lambda x: x[3])
                # Find the primary boundary (full canvas or highest z)
                primary = next(
                    (r for r in large_roots if r[1] == "spans full canvas"),
                    large_roots[-1]  # fallback to highest z
                )
                primary_z = primary[3]
                primary_rid = primary[0]
                primary_sp = next((s for s in ctx.subpaths if s.id == primary_rid), None)
                primary_area = primary_sp.area if primary_sp else 1.0
                descs = []
                for rid, ext, ch, z, shape in large_roots[:_MAX_APPENDAGES]:
                    z_label = f"z={z}" if z >= 0 else ""
                    if ext == "spans full canvas":
                        role = " ← primary boundary (outermost enclosing element)"
                    elif z < primary_z:
                        bg_sp = next((s for s in ctx.subpaths if s.id == rid), None)
                        area_pct = ""
                        if bg_sp and primary_area > 0:
                            ratio = bg_sp.area / primary_area * 100
                            if ratio > 100:
                                area_pct = f", area={ratio:.0f}% of primary (LARGER than primary boundary)"
                            else:
                                area_pct = f", area={ratio:.0f}% of primary"
                        shape_note = f", {shape}" if shape else ""
                        role = f" ← background layer (lower z-index, drawn first){shape_note}{area_pct}"
                    else:
                        role = ""
                    descs.append(f"{rid} ({ext}, {ch} children, {z_label}{role})")
                hints.append(
                    f"Major structural elements (top-level, ordered by draw layer — low z drawn first/behind, high z drawn last/in front): "
                    f"{'; '.join(descs)}."
                )

                # Background element layout
                bg_elements = [
                    (rid, ext, ch, z, shape)
                    for rid, ext, ch, z, shape in large_roots
                    if z < primary_z and ext != "spans full canvas"
                ]
                if bg_elements:
                    largest_bg = max(
                        bg_elements,
                        key=lambda x: next((s.area for s in ctx.subpaths if s.id == x[0]), 0),
                    )
                    lb_sp = next((s for s in ctx.subpaths if s.id == largest_bg[0]), None)
                    if lb_sp and ctx.canvas_height > 0 and ctx.canvas_width > 0:
                        lb_cy = lb_sp.centroid[1]
                        lb_y_pct = lb_cy / ctx.canvas_height * 100
                        lb_x1, lb_y1, lb_x2, lb_y2 = lb_sp.bbox
                        lb_top_pct = lb_y1 / ctx.canvas_height * 100
                        lb_bot_pct = lb_y2 / ctx.canvas_height * 100
                        lb_w_pct = (lb_x2 - lb_x1) / ctx.canvas_width * 100
                        lb_asp = lb_sp.features.get("aspect_ratio", 1.0)

                        top_sp = min(
                            (s for rid, *_ in large_roots if (s := next((sp for sp in ctx.subpaths if sp.id == rid), None))),
                            key=lambda s: s.centroid[1],
                        )
                        top_y_pct = top_sp.centroid[1] / ctx.canvas_height * 100
                        separation = abs(lb_y_pct - top_y_pct)

                        layout_parts = []
                        layout_parts.append(
                            f"Background element layout: largest background element ({largest_bg[0]}) "
                            f"centroid at y={lb_y_pct:.0f}%, bbox top edge at y={lb_top_pct:.0f}%, "
                            f"bbox bottom at y={lb_bot_pct:.0f}%"
                        )

                        if lb_top_pct < _QUAD_EMPTY_PCT:
                            layout_parts.append(
                                f"This background element extends UP into the upper half of the canvas "
                                f"(top edge at y={lb_top_pct:.0f}%)"
                            )

                        if lb_w_pct > _COVERAGE_INSCRIBED_CIRCLE:
                            layout_parts.append(
                                f"Width spans {lb_w_pct:.0f}% of canvas"
                            )

                        if lb_asp > _ASPECT_THRESHOLD:
                            layout_parts.append(
                                f"Aspect={lb_asp:.1f} (wider than tall) — spreads laterally"
                            )
                        elif lb_asp < _WAIST_NARROW_RATIO:
                            layout_parts.append(
                                f"Aspect={lb_asp:.1f} (taller than wide) — extends vertically"
                            )

                        if separation > _SEPARATION_NOTABLE_PCT:
                            if lb_y_pct > top_y_pct:
                                layout_parts.append(
                                    f"Top-most feature ({top_sp.id}) at y={top_y_pct:.0f}%. "
                                    f"Separation={separation:.0f}% — background element center is far below top feature"
                                )

                        hints.append(". ".join(layout_parts) + ".")

    # Cluster count
    if ctx.cluster_labels is not None:
        unique = set(int(l) for l in ctx.cluster_labels if l >= 0)
        if len(unique) > 2:
            hints.append(f"{len(unique)} spatial clusters.")

    # Shape variety
    shapes = {}
    for sp in ctx.subpaths:
        s = sp.features.get("shape_class", "organic")
        shapes[s] = shapes.get(s, 0) + 1
    if "circular" in shapes and shapes["circular"] >= 2:
        hints.append(f"{shapes['circular']} circular elements.")

    # Detail concentration — which side of the canvas has more SMALL elements
    # and concentric groups (tells the LLM where the "interesting" side is)
    if n > _DETAIL_ELEMENT_THRESHOLD:
        canvas_mid_x = ctx.canvas_width / 2
        left_detail = 0
        right_detail = 0
        for sp in ctx.subpaths:
            tier = sp.features.get("size_tier", "MEDIUM")
            cx = sp.centroid[0]
            if tier == "SMALL":
                if cx < canvas_mid_x:
                    left_detail += 1
                else:
                    right_detail += 1
        concentric = ctx.subpaths[0].features.get("concentric_groups", []) if ctx.subpaths else []
        for group in concentric:
            center = group.get("center", (0, 0))
            members = group.get("members", [])
            if len(members) >= 2:
                if center[0] < canvas_mid_x:
                    left_detail += 2
                else:
                    right_detail += 2
        total = left_detail + right_detail
        if total > 4:
            ratio = max(left_detail, right_detail) / total
            if ratio > _DETAIL_CONCENTRATION_RATIO:
                side = "right" if right_detail > left_detail else "left"
                hints.append(f"Detail/features concentrated on {side} side of canvas.")

    # Grid shape
    if grid:
        col_profile = _column_profile(grid)
        row_profile = _row_profile(grid)
        filled_cols = [i for i, v in enumerate(col_profile) if v > _FILL_OCCUPIED_MIN_PCT]
        if filled_cols:
            left_edge = filled_cols[0]
            right_edge = filled_cols[-1]
            span = right_edge - left_edge + 1
            total_cols = len(col_profile)
            if span < total_cols * _SILHOUETTE_SPAN_RATIO:
                side = "left" if left_edge < total_cols * _SILHOUETTE_NEAR_EDGE else ("right" if right_edge > total_cols * _SILHOUETTE_FAR_EDGE else "center")
                hints.append(f"Silhouette spans {span}/{total_cols} columns, toward {side}.")

    return hints


def _decompose_clusters(ctx: PipelineContext) -> list[str]:
    """Decompose the scene into cluster-based visual units."""
    if ctx.cluster_labels is None:
        return []

    labels = ctx.cluster_labels
    unique = sorted(set(int(l) for l in labels if l >= 0))
    if len(unique) < 2:
        return []

    canvas_w, canvas_h = ctx.canvas_width, ctx.canvas_height
    descriptions = []

    cluster_info = []
    for label in unique:
        indices = [i for i in range(len(labels)) if labels[i] == label]
        if len(indices) < _CLUSTER_MIN_MEMBERS:
            continue

        members = [ctx.subpaths[i] for i in indices]
        centroids_x = [sp.centroid[0] for sp in members]
        centroids_y = [sp.centroid[1] for sp in members]
        avg_x = sum(centroids_x) / len(centroids_x)
        avg_y = sum(centroids_y) / len(centroids_y)

        # Quadrant
        h_pos = "left" if avg_x < canvas_w * _THIRD else ("center" if avg_x < canvas_w * _TWO_THIRDS else "right")
        v_pos = "upper" if avg_y < canvas_h * _THIRD else ("mid" if avg_y < canvas_h * _TWO_THIRDS else "lower")

        # Dominant shape type
        shape_counts: dict[str, int] = {}
        for sp in members:
            s = sp.features.get("shape_class", "organic")
            shape_counts[s] = shape_counts.get(s, 0) + 1
        dominant_shape = max(shape_counts, key=shape_counts.get) if shape_counts else "mixed"

        # Total area
        total_area = sum(sp.area for sp in members)

        cluster_info.append({
            "label": label,
            "count": len(indices),
            "pos": f"{v_pos}-{h_pos}",
            "cx": avg_x,
            "cy": avg_y,
            "shape": dominant_shape,
            "area": total_area,
        })

        descriptions.append(
            f"Cluster {label} ({v_pos}-{h_pos}, {len(indices)} elements, "
            f"dominant shape: {dominant_shape})"
        )

    # Inter-cluster relationships — nearest neighbor per cluster only
    # (all-pairs is O(n²) lines for little LLM value)
    if len(cluster_info) >= 2:
        shown: set[tuple[int, int]] = set()
        for i in range(len(cluster_info)):
            best_j = -1
            best_dist = float("inf")
            ci = cluster_info[i]
            for j in range(len(cluster_info)):
                if j == i:
                    continue
                cj = cluster_info[j]
                dx = cj["cx"] - ci["cx"]
                dy = cj["cy"] - ci["cy"]
                d = (dx**2 + dy**2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0:
                pair = (min(i, best_j), max(i, best_j))
                if pair not in shown:
                    shown.add(pair)
                    cj = cluster_info[best_j]
                    descriptions.append(
                        f"Cluster {ci['label']} ({ci['pos']}) — "
                        f"Cluster {cj['label']} ({cj['pos']}) — "
                        f"separated by {best_dist:.0f}px"
                    )

    return descriptions


# ---------------------------------------------------------------------------
# Shape Narrative — contour walk, feature pairing, appendage classification
# ---------------------------------------------------------------------------


def _pos_label(x: float, y: float, w: float, h: float) -> str:
    """Return canvas-relative position label like 'upper-right'."""
    px = x / w if w else 0.5
    py = y / h if h else 0.5
    vl = "top" if py < _POS_QUARTER else ("upper" if py < _POS_INNER_LOW else ("mid" if py < _POS_INNER_HIGH else ("lower" if py < _POS_THREE_QUARTER else "bottom")))
    hl = "left" if px < _POS_QUARTER else ("center-left" if px < _POS_INNER_LOW else ("center" if px < _POS_INNER_HIGH else ("center-right" if px < _POS_THREE_QUARTER else "right")))
    if hl == "center" and vl == "mid":
        return "center"
    if hl.startswith("center"):
        return vl
    if vl == "mid":
        return hl
    return f"{vl}-{hl}"


def _compass8(angle_rad: float) -> str:
    """Convert angle (radians, math convention: 0=right, CCW) to 8-compass."""
    deg = math.degrees(angle_rad) % 360
    dirs = ["right", "upper-right", "up", "upper-left",
            "left", "lower-left", "down", "lower-right"]
    idx = int((deg + 22.5) / 45) % 8
    return dirs[idx]


def _signed_angle_diff(a: float, b: float) -> float:
    """Signed difference b - a, normalized to (-pi, pi]."""
    d = b - a
    while d > math.pi:
        d -= 2 * math.pi
    while d <= -math.pi:
        d += 2 * math.pi
    return d


_COMPASS_ORDER = ["right", "upper-right", "up", "upper-left",
                  "left", "lower-left", "down", "lower-right"]


def _compass_dist(a: str, b: str) -> int:
    """Distance between two compass directions (0-4, circular)."""
    ia = _COMPASS_ORDER.index(a)
    ib = _COMPASS_ORDER.index(b)
    d = abs(ia - ib)
    return min(d, 8 - d)


def _contour_walk(
    coords: list[tuple[float, float]], canvas_w: float, canvas_h: float
) -> str:
    """Walk clockwise around the contour and describe the shape progression.

    Merges adjacent segments aggressively to produce a readable ~8-12 segment
    description that captures the overall shape while noting sharp peaks and
    significant direction changes.
    """
    if len(coords) < 4:
        return ""

    diag = (canvas_w ** 2 + canvas_h ** 2) ** 0.5
    if diag == 0:
        return ""
    n = len(coords)

    # Start from vertex closest to bottom-center
    bc = (canvas_w / 2, canvas_h)
    start = min(range(n), key=lambda i: (coords[i][0] - bc[0]) ** 2 + (coords[i][1] - bc[1]) ** 2)
    ordered = coords[start:] + coords[:start]

    # Compute raw segments
    raw: list[dict] = []
    for i in range(n):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len = math.hypot(dx, dy)
        angle = math.atan2(-dy, dx)  # y-up convention

        if dy < -seg_len * _CONTOUR_RISE_FACTOR:
            verb = "rises"
        elif dy > seg_len * _CONTOUR_RISE_FACTOR:
            verb = "descends"
        else:
            verb = "extends"

        raw.append({
            "verb": verb,
            "dir": _compass8(angle),
            "angle": angle,
            "len": seg_len / diag,
            "start": p1,
            "end": p2,
        })

    # Detect significant peaks (sharp turn > 70 deg) and classify their width
    peak_at: set[int] = set()
    peak_width: dict[int, int] = {}  # peak_index -> number of segments in peak region
    for i in range(n):
        turn = abs(_signed_angle_diff(raw[i]["angle"], raw[(i + 1) % n]["angle"]))
        if turn > math.radians(_CONTOUR_PEAK_ANGLE_DEG):
            peak_at.add(i)
            # Compute peak width: count segments after peak that continue
            # in a reversed/perpendicular direction (>=2 compass steps
            # from approach means the contour has turned away)
            fwd_dir = raw[i]["dir"]
            rev_count = 0
            for k in range(1, min(8, n)):
                idx = (i + k) % n
                if _compass_dist(raw[idx]["dir"], fwd_dir) >= 2:
                    rev_count += 1
                else:
                    break
            peak_width[i] = rev_count

    # Aggressively merge: combine consecutive segments unless there's a peak
    # or the direction changes by more than 1 compass step
    merged: list[dict] = []
    i = 0
    while i < n:
        group_start = i
        total_len = raw[i]["len"]
        j = i + 1
        while j < n:
            if j in peak_at or (j - 1) in peak_at:
                break
            if _compass_dist(raw[j]["dir"], raw[group_start]["dir"]) > 1:
                break
            total_len += raw[j]["len"]
            j += 1

        # Compute merged segment properties
        start_pt = raw[group_start]["start"]
        end_pt = raw[(j - 1) % n]["end"]
        mid_x = (start_pt[0] + end_pt[0]) / 2
        mid_y = (start_pt[1] + end_pt[1]) / 2
        pos = _pos_label(mid_x, mid_y, canvas_w, canvas_h)
        is_peak = (j - 1) in peak_at
        pw = peak_width.get(j - 1, 0) if is_peak else 0

        # Dominant verb/direction from group
        verb = raw[group_start]["verb"]
        direction = raw[group_start]["dir"]

        merged.append({
            "verb": verb,
            "dir": direction,
            "pos": pos,
            "len": total_len,
            "peak": is_peak,
            "peak_width": pw,
            "count": j - group_start,
        })
        i = j

    # Build description text
    descs: list[str] = []
    for seg in merged:
        # Skip tiny segments unless they're peaks
        if seg["len"] < _CONTOUR_TINY_SEGMENT and not seg["peak"]:
            continue

        len_qual = ""
        if seg["len"] > _CONTOUR_LONG_SEGMENT:
            len_qual = "long "
        elif seg["len"] > _CONTOUR_MODERATE_SEGMENT:
            len_qual = ""  # normal
        elif seg["len"] > _CONTOUR_SHORT_SEGMENT:
            len_qual = "short "

        peak_note = ""
        if seg["peak"]:
            pw = seg.get("peak_width", 0)
            if pw >= _PEAK_WIDTH_BROAD:
                peak_note = " (broad curved lobe)"
            elif pw >= _PEAK_WIDTH_MODERATE:
                peak_note = " (wide curve)"
            else:
                peak_note = " (narrow peak)"

        if seg["count"] > _SEGMENT_MERGE_COUNT:
            text = f"{seg['verb']} through {seg['pos']} region ({len_qual}{seg['dir']}){peak_note}"
        else:
            text = f"{seg['verb']} {len_qual}{seg['dir']} at {seg['pos']}{peak_note}"

        descs.append(text)

    # Allow up to _MAX_CONTOUR_SEGMENTS for better shape coverage
    if len(descs) > _MAX_CONTOUR_SEGMENTS:
        descs = descs[:_MAX_CONTOUR_SEGMENTS] + ["...returns to start"]

    if not descs:
        return ""

    return "Starting from bottom-center: " + ", ".join(descs) + "."


def _detect_feature_pairs(ctx: PipelineContext) -> list[str]:
    """Detect y-aligned concentric group pairs and mirror-symmetric element pairs."""
    pairs: list[str] = []
    if not ctx.subpaths:
        return pairs

    canvas_w, canvas_h = ctx.canvas_width, ctx.canvas_height

    # Strategy 1: Concentric group y-alignment
    concentric = ctx.subpaths[0].features.get("concentric_groups", [])
    valid_groups: list[tuple[dict, tuple[float, float], float]] = []

    for group in concentric:
        center = group.get("center", (0.0, 0.0))
        members = group.get("members", [])
        if len(members) < 2:
            continue
        # Skip canvas-edge groups
        if (center[0] < canvas_w * _CANVAS_EDGE_MARGIN_PCT or center[0] > canvas_w * (1 - _CANVAS_EDGE_MARGIN_PCT)
                or center[1] < canvas_h * _CANVAS_EDGE_MARGIN_PCT or center[1] > canvas_h * (1 - _CANVAS_EDGE_MARGIN_PCT)):
            continue
        member_sps = [sp for sp in ctx.subpaths if sp.id in members]
        max_circ = max((sp.features.get("circularity", 0) for sp in member_sps), default=0)
        if max_circ > _CIRC_MODERATE:
            valid_groups.append((group, (float(center[0]), float(center[1])), max_circ))

    for i in range(len(valid_groups)):
        for j in range(i + 1, len(valid_groups)):
            g1, c1, circ1 = valid_groups[i]
            g2, c2, circ2 = valid_groups[j]
            y_diff = abs(c1[1] - c2[1])
            x_diff = abs(c1[0] - c2[0])
            if y_diff < canvas_h * _CENTER_TOLERANCE_PCT and x_diff > canvas_w * _CANVAS_EDGE_MARGIN_PCT:
                pos = _pos_label((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, canvas_w, canvas_h)
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                y_pct = ((c1[1] + c2[1]) / 2) / canvas_h * 100 if canvas_h else 0
                m1 = ",".join(g1["members"][:3])
                m2 = ",".join(g2["members"][:3])
                pairs.append(
                    f"Paired concentric groups at {pos}, y-aligned ~{y_pct:.0f}%, "
                    f"{dist:.0f}px apart ([{m1}] and [{m2}])"
                )

    # Strategy 2: Mirror pairs from global symmetry
    for pair_idx in ctx.symmetry_pairs:
        if len(pairs) >= 3:
            break
        a, b = pair_idx
        if a >= len(ctx.subpaths) or b >= len(ctx.subpaths):
            continue
        sp_a = ctx.subpaths[a]
        sp_b = ctx.subpaths[b]
        if sp_a.features.get("shape_class") != sp_b.features.get("shape_class"):
            continue
        # Only report structurally interesting pairs
        tier_a = sp_a.features.get("size_tier", "SMALL")
        tier_b = sp_b.features.get("size_tier", "SMALL")
        if tier_a == "SMALL" and tier_b == "SMALL":
            continue

        pos = _pos_label(
            (sp_a.centroid[0] + sp_b.centroid[0]) / 2,
            (sp_a.centroid[1] + sp_b.centroid[1]) / 2,
            canvas_w, canvas_h,
        )
        shape = sp_a.features.get("shape_class", "organic")
        circ = sp_a.features.get("circularity", 0)
        if circ > _CIRC_FEATURE_PAIR:
            pair_type = "circular"
        elif shape == "linear":
            pair_type = "linear"
        else:
            pair_type = shape
        axis = ctx.symmetry_axis or "vertical"
        pairs.append(
            f"Symmetric {pair_type} pair at {pos} ({sp_a.id}, {sp_b.id}), "
            f"mirrored across {axis} axis"
        )

    return pairs[:_MAX_FEATURE_PAIRS]


def _classify_appendages(
    ctx: PipelineContext, grid: list[list[bool]]
) -> list[str]:
    """Deprecated: merged into _detect_protrusions_from_elements().

    Kept for reference. Both functions detected the same elements using
    identical core logic. The protrusions version now includes shape type
    classification via _classify_shape_type().

    Original doc:
    Classify protruding elements + contour protrusions using computed features.
    Two strategies:
    1. Element-based: elements whose centroids are outside the core mass
    2. Contour-based: peaks in the contour vertices
    """
    results: list[str] = []

    if not ctx.subpaths or not grid:
        return results

    canvas_w, canvas_h = ctx.canvas_width, ctx.canvas_height
    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    if rows == 0 or cols == 0:
        return results

    row_fills = _row_profile(grid)
    col_fills = _column_profile(grid)

    core_rows = [i for i, v in enumerate(row_fills) if v > _QUAD_EMPTY_PCT]
    core_cols = [i for i, v in enumerate(col_fills) if v > _QUAD_EMPTY_PCT]
    if not core_rows or not core_cols:
        return results

    core_top = core_rows[0] / rows * canvas_h
    core_bottom = core_rows[-1] / rows * canvas_h
    core_left = core_cols[0] / cols * canvas_w
    core_right = core_cols[-1] / cols * canvas_w
    margin_x = canvas_w * _PROTRUSION_MARGIN_PCT
    margin_y = canvas_h * _PROTRUSION_MARGIN_PCT

    # --- Strategy 1: Element-based protrusions ---
    for sp in ctx.subpaths:
        cx, cy = sp.centroid
        direction = ""
        if cy < core_top - margin_y:
            direction = "above"
        elif cy > core_bottom + margin_y:
            direction = "below"
        elif cx < core_left - margin_x:
            direction = "left of"
        elif cx > core_right + margin_x:
            direction = "right of"
        else:
            continue

        tier = sp.features.get("size_tier", "MEDIUM")
        ar = sp.features.get("aspect_ratio", 1.0)
        circ = sp.features.get("circularity", 0.0)
        curv_mean = sp.features.get("curvature_mean", 0.0)
        curv_max = sp.features.get("curvature_max", 0.0)
        corners = sp.features.get("corner_count", 0)
        inflections = sp.features.get("inflection_count", 0)
        eccentricity = ar if ar > 1 else (1 / ar if ar > 0 else 1)

        shape_type = _classify_shape_type(eccentricity, circ, curv_mean, curv_max, corners, inflections)
        pos = _pos_label(cx, cy, canvas_w, canvas_h)
        desc = f"{sp.id}: {shape_type} {direction} core at {pos} [{tier}]"
        if eccentricity > _ECCENTRIC_MODERATE:
            desc += f", elongation={eccentricity:.1f}"
        results.append(desc)

    # --- Strategy 2: Contour-based protrusions (for complex SVGs) ---
    if len(results) < 2 and ctx.composite_grid is not None:
        from app.utils.rasterizer import extract_high_res_contour

        contour_coords = extract_high_res_contour(
            ctx.composite_silhouette, ctx.composite_grid,
            canvas_w, canvas_h, max_vertices=40,
        )
        if len(contour_coords) >= _MIN_CONTOUR_FOR_PROTRUSION:
            # Find centroid of contour
            cx_avg = sum(p[0] for p in contour_coords) / len(contour_coords)
            cy_avg = sum(p[1] for p in contour_coords) / len(contour_coords)
            # Compute distance from centroid for each vertex
            dists = [math.hypot(p[0] - cx_avg, p[1] - cy_avg) for p in contour_coords]
            # Tukey IQR fence: non-parametric, adapts to data distribution
            dists_arr = np.array(dists)
            q1, q3 = float(np.percentile(dists_arr, 25)), float(np.percentile(dists_arr, 75))
            threshold = q3 + _TUKEY_K * (q3 - q1)
            for idx, (pt, d) in enumerate(zip(contour_coords, dists)):
                if d > threshold:
                    # Check if this is a local peak (higher than neighbors)
                    prev_d = dists[(idx - 1) % len(dists)]
                    next_d = dists[(idx + 1) % len(dists)]
                    if d >= prev_d and d >= next_d:
                        angle = math.atan2(-(pt[1] - cy_avg), pt[0] - cx_avg)
                        direction = _compass8(angle)
                        pos = _pos_label(pt[0], pt[1], canvas_w, canvas_h)
                        prominence = (d - mean_dist) / mean_dist * 100
                        results.append(
                            f"Contour protrusion toward {direction} at {pos}, "
                            f"prominence={prominence:.0f}% above mean radius"
                        )

    # Sort element-based by area, limit
    sp_map = {sp.id: sp.area for sp in ctx.subpaths}
    results.sort(key=lambda r: sp_map.get(r.split(":")[0], 0) if ":" in r else 0, reverse=True)
    return results[:_MAX_APPENDAGES]


def _classify_shape_type(
    eccentricity: float, circ: float, curv_mean: float, curv_max: float,
    corners: int, inflections: int,
) -> str:
    """Classify an element's shape using generic geometric terms."""
    if eccentricity > _ECCENTRIC_HIGH and curv_mean < _DIRECTION_TOLERANCE_PCT:
        return "thin straight extension"
    if eccentricity > _ECCENTRIC_HIGH and curv_mean >= _DIRECTION_TOLERANCE_PCT:
        return "thin curved extension"
    if eccentricity > _ECCENTRIC_MODERATE and curv_max > _CIRC_MODERATE:
        return "narrow pointed extension"
    if circ > _CIRC_LOBE_SHAPE:
        return "rounded lobe"
    if corners >= _ANGULAR_CORNERS:
        return "angular extension"
    if inflections >= _SINUOUS_INFLECTIONS:
        return "sinuous extension"
    if eccentricity > _ELONGATED_ECCENTRICITY:
        return "elongated extension"
    return "organic extension"


def _surface_key_features(ctx: PipelineContext) -> list[str]:
    """Surface high-value unused features for top elements.

    Only shows features that differentiate elements — skips properties
    that are uniform across all elements (e.g., same role, same gap %).
    """
    if not ctx.subpaths:
        return []

    sorted_sps = sorted(ctx.subpaths, key=lambda sp: sp.area, reverse=True)
    top_sps = sorted_sps[:_TOP_FEATURES_N]

    # Compute feature distributions to find what's uniform vs varied
    purposes = [sp.features.get("construction_purpose", "") for sp in top_sps]
    gap_pcts = [sp.features.get("total_gap_area_pct", 0) for sp in top_sps]
    purpose_uniform = len(set(p for p in purposes if p)) <= 1
    gap_uniform = (max(gap_pcts) - min(gap_pcts)) < _GAP_UNIFORMITY_PCT if gap_pcts else True

    results: list[str] = []

    for sp in top_sps:
        parts: list[str] = []

        # Construction purpose (T3.13) — only if varied across elements
        if not purpose_uniform:
            purpose = sp.features.get("construction_purpose", "")
            if purpose and purpose not in ("unknown", "single_element", ""):
                parts.append(f"role={purpose}")

        # Curvature character (T1.01)
        curv_mean = sp.features.get("curvature_mean", 0)
        if curv_mean > _CONVEX_LOW:
            parts.append("highly curved")
        elif curv_mean > _DIRECTION_TOLERANCE_PCT:
            parts.append("moderately curved")
        elif curv_mean > _CURVATURE_NEGLIGIBLE:
            parts.append("mostly straight edges")

        # Inflection complexity (T1.02)
        inflections = sp.features.get("inflection_count", 0)
        if inflections >= _INFLECTIONS_COMPLEX:
            parts.append(f"complex boundary ({inflections} inflections)")
        elif inflections >= _INFLECTIONS_MODERATE:
            parts.append(f"moderate complexity ({inflections} inflections)")

        # Containment depth (T3.01)
        depth = sp.features.get("containment_depth", 0)
        if depth >= 2:
            parts.append(f"deeply nested (depth={depth})")

        # Medial axis topology (T1.12)
        junctions = sp.features.get("medial_axis_junctions", 0)
        endpoints = sp.features.get("medial_axis_endpoints", 0)
        if junctions >= 2:
            parts.append(f"branching internal structure ({junctions} junctions, {endpoints} endpoints)")
        elif endpoints == 2 and junctions == 0:
            parts.append("simple rod-like internal structure")

        # Gap structure (T3.12) — only if varied across elements
        if not gap_uniform:
            gap_pct = sp.features.get("total_gap_area_pct", 0)
            gap_pattern = sp.features.get("gap_pattern", "none")
            if gap_pct > _GAP_REPORT_MIN_PCT:
                parts.append(f"internal gaps ({gap_pct:.0f}% area, pattern={gap_pattern})")

        if parts:
            results.append(f"{sp.id}: {', '.join(parts)}")

    return results[:_MAX_FEATURES]


def _find_primary_boundary(ctx: PipelineContext):
    """Find the primary boundary element — the outermost shape that defines the figure.

    For complex illustrations this is the element containing the most others.
    For simple SVGs, returns None (use composite silhouette instead).
    """
    if len(ctx.subpaths) < _MIN_ELEMENTS_FOR_PRIMARY:
        return None

    best_sp = None
    best_children = 0
    for sp in ctx.subpaths:
        n_children = len(sp.features.get("contains", []))
        if n_children > best_children:
            best_children = n_children
            best_sp = sp

    # Only use primary boundary if it contains a meaningful fraction of elements
    if best_sp and best_children >= len(ctx.subpaths) * _PRIMARY_CHILD_RATIO:
        return best_sp
    return None


def _polygon_to_contour(
    polygon, canvas_w: float, canvas_h: float, max_vertices: int = 40,
) -> list[tuple[float, float]]:
    """Extract simplified contour coordinates from a Shapely polygon.

    For MultiPolygon (complex paths fragmented into pieces), rasterize
    all geoms to a grid and extract the contour from that — this captures
    the complete shape outline rather than just the largest fragment.
    """
    if polygon is None or polygon.is_empty:
        return []

    from shapely.geometry import MultiPolygon

    def _ensure_cw(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        s = 0.0
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            s += (x2 - x1) * (y2 + y1)
        return list(reversed(pts)) if s < 0 else pts

    # For MultiPolygon, use the rasterized grid to extract a unified contour
    if isinstance(polygon, MultiPolygon):
        from app.utils.rasterizer import rasterize_polygon_to_grid, extract_high_res_contour

        grid = rasterize_polygon_to_grid(polygon, canvas_w, canvas_h, resolution=_SILHOUETTE_GRID_RES)
        if np.sum(grid) >= _MIN_FILLED_GRID_CONTOUR:
            return extract_high_res_contour(
                None, grid, canvas_w, canvas_h, max_vertices=max_vertices
            )
        return []

    if polygon.exterior is None:
        return []

    coords = list(polygon.exterior.coords)
    if len(coords) > max_vertices:
        lo, hi = 0.0, max(canvas_w, canvas_h) * _CONVEX_LOW
        best = coords
        for _ in range(_SIMPLIFICATION_ITERS):
            mid = (lo + hi) / 2
            simplified = polygon.simplify(mid)
            if simplified.is_empty or simplified.exterior is None:
                hi = mid
                continue
            sc = list(simplified.exterior.coords)
            if len(sc) <= max_vertices:
                best = sc
                hi = mid
            else:
                lo = mid
        coords = best
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return []
    return _ensure_cw([(float(x), float(y)) for x, y in coords])


def _build_shape_narrative(
    ctx: PipelineContext, grid: list[list[bool]]
) -> ShapeNarrative:
    """Build shape narrative by synthesizing existing geometric features."""
    contour_walk = ""

    # For complex illustrations, walk the primary boundary element's actual shape
    primary = _find_primary_boundary(ctx)
    if primary is not None and primary.polygon is not None:
        contour_coords = _polygon_to_contour(
            primary.polygon, ctx.canvas_width, ctx.canvas_height, max_vertices=40,
        )
        if contour_coords:
            contour_walk = _contour_walk(contour_coords, ctx.canvas_width, ctx.canvas_height)

    # Fall back to composite grid contour for simple SVGs
    if not contour_walk and ctx.composite_grid is not None:
        from app.utils.rasterizer import extract_high_res_contour

        contour_coords = extract_high_res_contour(
            ctx.composite_silhouette,
            ctx.composite_grid,
            ctx.canvas_width,
            ctx.canvas_height,
            max_vertices=40,
        )
        if contour_coords:
            contour_walk = _contour_walk(contour_coords, ctx.canvas_width, ctx.canvas_height)

    feature_pairs = _detect_feature_pairs(ctx)
    # Appendages merged into protrusions (SpatialInterpretation.protrusions)
    # to eliminate duplication — both detected the same elements.
    appendages: list[str] = []
    detail_features = _surface_key_features(ctx)

    return ShapeNarrative(
        contour_walk=contour_walk,
        feature_pairs=feature_pairs,
        appendages=appendages,
        detail_features=detail_features,
    )

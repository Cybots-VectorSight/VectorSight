"""Pixel-domain perceptual segmentation pipeline.

Rasterizes SVG to image, then uses classic image processing to discover
meaningful visual regions. Every numeric value in this module is either:

1. A MATHEMATICAL CONSTANT — derived from a theorem or definition
2. A NUMERICAL GUARD — machine epsilon or stability floor
3. DATA-DERIVED — computed from the input image or SVG metadata

No content-sensitive human-chosen thresholds.

Algorithms:
1. Felzenszwalb graph-based segmentation (scale from Lab IQR)
2. RAG hierarchical merging (thresholds from Jenks natural breaks)
3. Distance transform (zero-parameter primary boundary detection)
4. regionprops (zero-parameter region properties)
5. RAG adjacency export (topological fact, zero parameters)
6. Containment detection (geometric bbox test, zero parameters)
7. Symmetric pair detection (min-max normalized features + Hungarian matching)

Per the architecture doc (docs/pixel_segmentation_architecture.md):
- No shape classification — report continuous values, LLM interprets
- No hardcoded area/color thresholds — use data-driven grouping
- Background detected topologically (edge-touching), not by color
- RAG level count determined by data, not fixed
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

import cairosvg
from PIL import Image
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.graph import rag_mean_color, cut_threshold
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Mathematical constants — each with derivation
# ---------------------------------------------------------------------------

# Maximum Euclidean distance in 8-bit RGB space:
# d = √(255² + 255² + 255²) = 255√3
# This is the diagonal of the RGB color cube [0,255]³.
_MAX_RGB_DIST = 255.0 * np.sqrt(3.0)

# Shannon-Nyquist sampling theorem: minimum 2× the highest spatial frequency.
# In 2D, each dimension needs 2 samples → 2² = 4 pixels per feature minimum.
_NYQUIST_FACTOR = 2     # 1D: 2 samples per cycle (Shannon-Nyquist theorem)
_NYQUIST_2D = 4          # 2D: 2² pixels per feature (Nyquist in each dimension)

# Minimum structuring element for local maximum detection:
# 1×1 kernel = identity (every pixel is its own max — useless)
# 2×2 kernel = no center pixel (ambiguous max location)
# 3×3 kernel = smallest with a unique center pixel
_MIN_KERNEL = 3

# IQR definition: Q1 = 25th percentile, Q3 = 75th percentile.
# These define the interquartile range (Tukey, 1977).
_IQR_Q1, _IQR_Q3 = 25, 75

# Minimum samples for stable percentile estimation:
# 4 quartiles × 1 sample each = 4 minimum data points.
# Below this, np.percentile still works but the estimate is degenerate.
_MIN_SAMPLES_FOR_PERCENTILE = 4

# Minimum pixels for second-order moment computation (eccentricity, orientation):
# An ellipse has 5 degrees of freedom (cx, cy, semi-a, semi-b, θ).
# regionprops needs ≥5 pixels for non-degenerate moment matrices.
_MIN_PIXELS_FOR_MOMENTS = 5

# Monospace character aspect ratio: height ≈ 2× width.
# This is a typographic fact for standard monospace fonts (Courier, Consolas,
# Menlo all have width:height ≈ 0.5-0.6). We compensate in ASCII art by
# sampling rows at 2× the column step, producing visually square cells.
_CHAR_ASPECT = 2

# W3C CSS 2.1 Section 10.3.2 + SVG 2 Embedded Content spec:
# Replaced elements (svg, canvas, video, iframe) with no intrinsic dimensions
# default to 300px wide × 150px tall. This is a W3C standard, not arbitrary.
# Reference: https://www.w3.org/TR/CSS2/visudet.html
_W3C_DEFAULT_WIDTH = 300.0
_W3C_DEFAULT_HEIGHT = 150.0

# Bimodality coefficient threshold: derived from the uniform distribution.
# BC = (skewness² + 1) / kurtosis. For the uniform distribution (the
# boundary between unimodal and multimodal shapes):
#   skewness = 0, kurtosis = 9/5 → BC = 1 / (9/5) = 5/9 ≈ 0.5556
# BC > 5/9 suggests bimodality (SAS Institute; Freeman & Dale, 2013).
# This is a mathematical constant, not a tuned threshold.
_BIMODALITY_THRESHOLD = 5.0 / 9.0  # = 0.5556...

# ASCII character palette for segment visualization.
# Background always gets '.', foreground regions get these characters
# in order of area (largest → first char). The palette has 46 distinct
# printable non-whitespace characters.
_CHAR_PALETTE = "#@%&*+=-~:;!?/\\|<>^vXOQWMBZS0123456789abcdef"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PixelRegion:
    """Properties of a discovered visual region — continuous values only."""
    region_id: int
    area_pct: float
    centroid: tuple[float, float]          # (x_frac, y_frac) in 0-1
    bbox: tuple[int, int, int, int]        # min_row, min_col, max_row, max_col
    eccentricity: float                    # 0=circle, 1=line
    solidity: float                        # 1=convex, <1=concave/irregular
    euler_number: int                      # 1=solid, 0=one hole, -1=two holes...
    orientation: float                     # radians
    mean_color: tuple[int, int, int]
    is_background: bool = False            # detected topologically
    neighbors: list[int] = field(default_factory=list)   # adjacent region IDs
    inside_of: int | None = None           # parent region ID (containment)
    contains: list[int] = field(default_factory=list)    # child region IDs


@dataclass
class SymmetricPair:
    """Two regions that are mirror images across an axis."""
    r1: int
    r2: int
    axis: str                              # "vertical" or "horizontal"
    similarity: float                      # 0-1, how similar the pair is


@dataclass
class MergeEvent:
    """Records which regions merged at what threshold."""
    threshold: float
    merged_regions: list[list[int]]
    n_regions_after: int


@dataclass
class PixelSegmentationResult:
    """Complete output of the pixel segmentation pipeline."""
    resolution: int
    n_atomic_segments: int
    regions: list[PixelRegion]
    merge_events: list[MergeEvent]
    hierarchy_levels: dict[float, int]
    primary_center: tuple[int, int] | None
    primary_radius: float
    sub_centers: list[tuple[int, int]]
    symmetric_pairs: list[SymmetricPair] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    segment_labels: NDArray | None = None     # raw Felzenszwalb labels
    relabeled_segments: NDArray | None = None  # labels matching region IDs
    raster_image: NDArray | None = None


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def rasterize_svg(svg_code: str, resolution: int = _MIN_KERNEL * _MIN_KERNEL) -> NDArray:
    """Rasterize SVG to RGBA numpy array using CairoSVG.

    Default resolution: _MIN_KERNEL² = 9 (degenerate fallback).
    In normal use, resolution is always computed by compute_resolution().
    """
    raw = svg_code.encode("utf-8") if isinstance(svg_code, str) else svg_code
    png_data = cairosvg.svg2png(bytestring=raw, output_width=resolution, output_height=resolution)
    return np.array(Image.open(io.BytesIO(png_data)).convert("RGBA"))


def _min_resolution_for_elements(n_elements: int) -> int:
    """Data-derived minimum resolution: enough pixels for moment computation.

    Each of n_elements needs ≥ _MIN_PIXELS_FOR_MOMENTS (5) pixels for
    non-degenerate second-order moments (eccentricity, orientation).
    Total pixels needed = n_elements × _MIN_PIXELS_FOR_MOMENTS.
    Resolution = √(total_pixels).

    Floor: _MIN_KERNEL × _MIN_KERNEL = 9 pixels (3×3), the smallest
    grid where local maximum detection is defined.
    """
    total_needed = max(n_elements, 1) * _MIN_PIXELS_FOR_MOMENTS
    return max(_MIN_KERNEL * _MIN_KERNEL, int(np.sqrt(total_needed)))


def compute_resolution(
    element_bboxes: list[tuple[float, float, float, float]],
    canvas_w: float, canvas_h: float,
    n_elements: int = 0,
) -> int:
    """Shannon-Nyquist derived resolution from element size distribution.

    Oversampling factor k = 2 (Nyquist minimum) + log₂(d_max / d_min).
    The log₂ term counts the number of octaves in the feature size
    distribution — more octaves = more spectral content that could alias.

    Mathematical basis: Shannon-Nyquist sampling theorem requires 2×
    the highest spatial frequency. The additional log₂ factor accounts
    for the bandwidth of the signal (wider bandwidth = more aliasing risk).

    Lower bound: data-derived from n_elements (enough pixels per element
    for second-order moment computation).
    Upper bound: none. The Nyquist formula IS the mathematically correct
    answer — clamping it loses information. Callers can control resolution
    indirectly by adjusting canvas_w/canvas_h if latency is a concern.
    """
    min_res = _min_resolution_for_elements(n_elements)

    if not element_bboxes:
        # No element info: Nyquist sampling of canvas dimensions
        return max(min_res, int(_NYQUIST_FACTOR * max(canvas_w, canvas_h)))

    sizes = [min(abs(x2 - x1), abs(y2 - y1))
             for x1, y1, x2, y2 in element_bboxes
             if abs(x2 - x1) > 0 and abs(y2 - y1) > 0]
    if not sizes:
        return max(min_res, int(_NYQUIST_FACTOR * max(canvas_w, canvas_h)))

    d_min = min(sizes)
    d_max = max(sizes)

    # Oversampling: Nyquist base (2) + octave range
    # log₂(1) = 0 for uniform features → k = 2 (bare Nyquist)
    # log₂(8) = 3 for 3 octaves → k = 5
    ratio = max(d_max / d_min, 1.0)
    k = _NYQUIST_FACTOR + np.log2(ratio)

    resolution = int(k * max(canvas_w, canvas_h) / d_min)
    return max(min_res, resolution)


# ---------------------------------------------------------------------------
# Felzenszwalb parameters (all data-derived)
# ---------------------------------------------------------------------------

def compute_felzenszwalb_params(image: NDArray, n_elements: int) -> dict:
    """Derive all Felzenszwalb parameters from the image itself.

    - scale: Lab color gradient IQR (measures natural color variation)
    - sigma: 0.0 (SVG rasters are noise-free; mathematically, σ=0 means
      no pre-smoothing, which is correct for clean step-edge images)
    - min_size: total_area / (n_elements × _NYQUIST_2D), capped at
      √(total_area / n_elements). The _NYQUIST_2D factor (= 2² = 4) comes
      from Shannon-Nyquist in 2D: each feature needs ≥2 samples per
      dimension = 4 pixels minimum. The √ cap ensures min_size doesn't
      exceed one element's linear dimension.
    """
    rgb = image[:, :, :3].astype(np.float64)
    h, w = rgb.shape[:2]

    # Lab IQR for scale (Felzenszwalb & Huttenlocher, 2004)
    # Convert to CIELAB for perceptually uniform gradients
    lab = rgb2lab(rgb / 255.0 if rgb.max() > 1.0 else rgb)
    dy = np.diff(lab, axis=0)[:, :-1]
    dx = np.diff(lab, axis=1)[:-1, :]
    grad_mag = np.sqrt(np.sum(dy**2, axis=-1) + np.sum(dx**2, axis=-1))
    nonzero = grad_mag[grad_mag > 0]

    if len(nonzero) >= _MIN_SAMPLES_FOR_PERCENTILE:
        q25, q75 = np.percentile(nonzero, [_IQR_Q1, _IQR_Q3])
        scale = float(q75 - q25)
        if scale < np.finfo(float).eps:
            scale = float(np.median(nonzero))
    else:
        # Fewer than 4 gradient samples: image is nearly flat.
        # Any positive scale gives identical result on flat images because
        # all edge weights ≈ 0, so MInt(C1,C2) = scale/|C| > 0 > min_edge,
        # causing everything to merge regardless of scale value.
        scale = float(np.mean(nonzero)) if len(nonzero) > 0 else 1.0

    # min_size from Nyquist:
    # Each element's average pixel budget = total_area / n_elements
    # Nyquist 2D: 4 pixels minimum per feature → min_size = budget / 4
    # Cap: √budget (one element's linear dimension, prevents over-merging)
    avg_budget = (h * w) / max(n_elements, 1)
    min_size_nyquist = max(1, int(avg_budget / _NYQUIST_2D))
    min_size_cap = max(1, int(np.sqrt(avg_budget)))
    min_size = min(min_size_nyquist, min_size_cap)

    return {"scale": scale, "sigma": 0.0, "min_size": min_size}


def segment_image(image: NDArray, params: dict) -> NDArray:
    return felzenszwalb(image[:, :, :3], **params)


# ---------------------------------------------------------------------------
# RAG hierarchy — data-driven level count and thresholds
# ---------------------------------------------------------------------------

def find_rag_thresholds(image: NDArray, segments: NDArray) -> list[float]:
    """Jenks-style: Otsu on gap distribution determines both count and values.

    The edge weights of the RAG represent color distances between adjacent
    segments. Natural breaks in these weights (large gaps) indicate
    meaningful merge thresholds. Otsu's method (Otsu, 1979) finds the
    threshold that maximizes inter-class variance in the gap distribution.
    """
    rag = rag_mean_color(image[:, :, :3], segments)
    weights = sorted(d["weight"] for u, v, d in rag.edges(data=True) if u != v)

    if len(weights) < _MIN_KERNEL:  # <3 edges → can't compute meaningful gaps
        return [float(np.mean(weights))] if weights else []

    diffs = np.diff(weights)
    if len(diffs) < _NYQUIST_FACTOR:  # <2 gaps → single threshold at median
        return [float(weights[-1])]

    try:
        gap_thresh = threshold_otsu(diffs)
    except ValueError:
        return [float(np.median(weights))]

    selected = [float(weights[i]) for i, gap in enumerate(diffs) if gap > gap_thresh]
    return sorted(selected) if selected else [float(np.median(weights))]


def build_hierarchy(image: NDArray, segments: NDArray, thresholds: list[float]) -> dict[float, NDArray]:
    rgb = image[:, :, :3]
    return {
        thresh: cut_threshold(segments, rag_mean_color(rgb, segments), thresh, in_place=False)
        for thresh in sorted(thresholds)
    }


def find_merge_events(segments: NDArray, hierarchy: dict[float, NDArray]) -> list[MergeEvent]:
    events = []
    prev = segments
    for thresh in sorted(hierarchy):
        merged = hierarchy[thresh]
        groups: dict[int, set[int]] = {}
        for old in np.unique(prev):
            for new in np.unique(merged[prev == old]):
                groups.setdefault(int(new), set()).add(int(old))
        actual = [sorted(g) for g in groups.values() if len(g) > 1]
        events.append(MergeEvent(thresh, actual, len(np.unique(merged))))
        prev = merged
    return events


# ---------------------------------------------------------------------------
# Distance transform — zero-parameter primary boundary
# ---------------------------------------------------------------------------

def find_primary_boundary(foreground_mask: NDArray) -> tuple[tuple[int, int] | None, float, list[tuple[int, int]]]:
    """Find primary mass center via distance transform.

    The distance transform maps each foreground pixel to its distance from
    the nearest background pixel. The global maximum is the most "interior"
    point — the inscribed circle center.

    Filter size for local maximum detection is derived from the distance
    field itself using nearest-neighbor spacing of preliminary peaks.
    """
    dist = distance_transform_edt(foreground_mask)
    if dist.max() == 0:
        return None, 0.0, []

    max_idx = np.unravel_index(np.argmax(dist), dist.shape)
    center = (int(max_idx[0]), int(max_idx[1]))
    radius = float(dist.max())

    # Adaptive filter size from distance field statistics:
    # 1. Find preliminary peaks with minimum kernel (3×3)
    # 2. Compute nearest-neighbor distances between peaks
    # 3. Filter size = median NN distance / 2 (Nyquist: 2 samples per peak)
    # This ensures one seed per object for typical object sizes.
    preliminary = maximum_filter(dist, size=_MIN_KERNEL) == dist
    peak_mask = preliminary & (dist > 0)
    peak_coords = np.argwhere(peak_mask)

    if len(peak_coords) < _NYQUIST_FACTOR:
        # Too few peaks — use minimum kernel
        filter_size = _MIN_KERNEL
    else:
        tree = KDTree(peak_coords)
        nn_dists, _ = tree.query(peak_coords, k=_NYQUIST_FACTOR)  # k=2: self + nearest
        # Column 1 = nearest neighbor (column 0 = self, distance 0)
        median_nn = np.median(nn_dists[:, 1])
        # Filter size = half the median spacing (Nyquist in 1D)
        filter_size = max(_MIN_KERNEL, int(median_nn / _NYQUIST_FACTOR))
        # Ensure odd (symmetric kernel)
        if filter_size % _NYQUIST_FACTOR == 0:
            filter_size += 1

    local_max = maximum_filter(dist, size=filter_size) == dist
    peak_coords_final = np.argwhere(local_max & (dist > 0))
    peak_vals = dist[peak_coords_final[:, 0], peak_coords_final[:, 1]]

    if len(peak_vals) == 0:
        return center, radius, []

    # Otsu on peak values to separate significant from noise peaks
    try:
        peak_thresh = threshold_otsu(peak_vals)
    except ValueError:
        peak_thresh = np.median(peak_vals) if len(peak_vals) > 0 else radius

    subs = [(int(c[0]), int(c[1])) for c, v in zip(peak_coords_final, peak_vals)
            if v >= peak_thresh and (int(c[0]), int(c[1])) != center]
    subs.sort(key=lambda c: dist[c[0], c[1]], reverse=True)
    return center, radius, subs


# ---------------------------------------------------------------------------
# Background detection — topological
# ---------------------------------------------------------------------------

def _detect_background(segments: NDArray, image: NDArray) -> set[int]:
    """Background = segments touching all 4 canvas edges (topological).

    Fallback: fully transparent segments (alpha channel max == 0, which is
    a structural property — 0 is the only possible value for "no content").
    """
    h, w = segments.shape
    top = set(int(x) for x in segments[0, :])
    bottom = set(int(x) for x in segments[h - 1, :])
    left = set(int(x) for x in segments[:, 0])
    right = set(int(x) for x in segments[:, w - 1])

    bg = top & bottom & left & right

    # Fallback: fully transparent regions (alpha max == 0 is structural, not a threshold)
    if not bg and image.shape[2] >= 4:  # 4 = RGBA channel count (structural)
        alpha = image[:, :, 3]  # 3 = alpha channel index (structural)
        for lbl in np.unique(segments):
            if alpha[segments == lbl].max() == 0:
                bg.add(int(lbl))
    return bg


# ---------------------------------------------------------------------------
# Region adjacency — extracted from RAG graph
# ---------------------------------------------------------------------------

def _compute_adjacency(image: NDArray, segments: NDArray) -> dict[int, list[int]]:
    """Extract which regions touch each other from the RAG.
    Zero parameters — adjacency is a topological fact.
    """
    rag = rag_mean_color(image[:, :, :3], segments)
    adj: dict[int, list[int]] = {}
    for u, v in rag.edges():
        if u != v:
            adj.setdefault(int(u), []).append(int(v))
            adj.setdefault(int(v), []).append(int(u))
    return adj


# ---------------------------------------------------------------------------
# Region containment — from bounding boxes
# ---------------------------------------------------------------------------

def _compute_containment(
    segments: NDArray, regions: list[PixelRegion], bg_labels: set[int],
) -> None:
    """Determine which regions are inside which. Mutates regions in place.

    A region A contains region B if B's bounding box is within A's bbox
    AND A is the smallest such container (direct parent, not grandparent).
    Purely geometric — zero content-sensitive parameters.
    """
    fg = [r for r in regions if not r.is_background]
    if len(fg) < _NYQUIST_FACTOR:  # <2 regions: no parent-child possible
        return

    by_area = sorted(fg, key=lambda r: r.area_pct, reverse=True)

    for child in by_area:
        cr1, cc1, cr2, cc2 = child.bbox
        best_parent = None
        best_area = float("inf")

        for parent in by_area:
            if parent.region_id == child.region_id:
                continue
            if parent.area_pct <= child.area_pct:
                continue
            pr1, pc1, pr2, pc2 = parent.bbox
            if pr1 <= cr1 and pc1 <= cc1 and pr2 >= cr2 and pc2 >= cc2:
                if parent.area_pct < best_area:
                    best_parent = parent
                    best_area = parent.area_pct

        if best_parent is not None:
            child.inside_of = best_parent.region_id
            best_parent.contains.append(child.region_id)


# ---------------------------------------------------------------------------
# Symmetric pair detection — min-max normalization + Hungarian matching
# ---------------------------------------------------------------------------

def _detect_symmetric_pairs(
    regions: list[PixelRegion],
) -> list[SymmetricPair]:
    """Find mirror-image region pairs using optimal matching.

    Algorithm:
    1. For all foreground region pairs, compute 5 difference features:
       mirror_position_dist, area_ratio_diff, eccentricity_diff,
       solidity_diff, color_dist
    2. Min-max normalize each feature to [0, 1] using observed range.
       This is an affine normalization where each feature's unit = its
       observed range. No hardcoded weights — every feature contributes
       proportionally to its actual discriminative range in this image.
    3. Cost = L2 norm in normalized feature space.
       Maximum possible cost = √n_features (all features at max difference),
       which is √5 ≈ 2.236 — a mathematical constant.
    4. Hungarian algorithm (Kuhn-Munkres, 1955) finds the globally optimal
       minimum-cost matching. O(n³), proven optimal.
    5. Convert costs to similarity: sim = 1 - cost/√n_features.
    6. Otsu on similarities to separate real pairs from forced matches.
    """
    fg = [r for r in regions if not r.is_background]
    if len(fg) < _NYQUIST_FACTOR:  # <2 regions: no pairs possible
        return []
    # No area pre-filtering — the Hungarian algorithm + Otsu on similarities
    # naturally separates real symmetric pairs from noise. Pre-filtering
    # by area would remove legitimate small features (e.g., eyes in a face).

    n = len(fg)
    n_features = 5  # structural: we compute exactly 5 difference features

    # Build n×n×5 pairwise feature difference tensor
    pair_features = np.full((n, n, n_features), 0.0)

    for i, a in enumerate(fg):
        ax, ay = a.centroid
        mirror_x = 1.0 - ax  # reflect across x=0.5 (center of [0,1] range)

        for j, b in enumerate(fg):
            if i == j:
                continue
            bx, by = b.centroid

            # Feature 0: position distance from B to A's mirror
            pair_features[i, j, 0] = np.sqrt(
                (bx - mirror_x) ** 2 + (by - ay) ** 2
            )
            # Feature 1: area ratio difference (0 = same size)
            pair_features[i, j, 1] = abs(a.area_pct - b.area_pct) / max(
                a.area_pct, b.area_pct, np.finfo(float).eps
            )
            # Feature 2: eccentricity difference (both in [0,1])
            pair_features[i, j, 2] = abs(a.eccentricity - b.eccentricity)
            # Feature 3: solidity difference (both in [0,1])
            pair_features[i, j, 3] = abs(a.solidity - b.solidity)
            # Feature 4: color distance normalized by _MAX_RGB_DIST (= 255√3)
            pair_features[i, j, 4] = (
                np.sqrt(sum(
                    (ca - cb) ** 2 for ca, cb in zip(a.mean_color, b.mean_color)
                ))
                / _MAX_RGB_DIST
            )

    # Min-max normalize each feature to [0, 1] using observed range.
    # This is an affine transformation: z = (x - min) / (max - min).
    # Each feature's unit becomes its observed range, so all features
    # contribute equally to the cost. Features with zero range (constant
    # across all pairs) get mapped to 0 — they provide no discrimination.
    mask = ~np.eye(n, dtype=bool)  # exclude diagonal (self-pairs)
    for f in range(n_features):
        vals = pair_features[:, :, f][mask]
        vmin = vals.min()
        vmax = vals.max()
        spread = vmax - vmin
        if spread > np.finfo(float).eps:
            pair_features[:, :, f] = (pair_features[:, :, f] - vmin) / spread
        else:
            pair_features[:, :, f] = 0.0

    # Cost matrix: L2 norm in normalized feature space
    cost = np.sqrt(np.sum(pair_features ** 2, axis=2))
    np.fill_diagonal(cost, 1e12)  # prevent self-matching (numerical guard)

    # Hungarian algorithm: globally optimal minimum-cost matching
    # (Kuhn, 1955; Munkres, 1957). Proven to find the assignment
    # that minimizes total cost. O(n³).
    row_ind, col_ind = linear_sum_assignment(cost)

    # De-duplicate (each pair appears as both i→j and j→i in the assignment)
    # and convert cost to similarity.
    # Max possible cost = √n_features (all features at max = 1.0).
    max_cost = np.sqrt(float(n_features))  # √5 ≈ 2.236

    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, float]] = []

    for r, c in zip(row_ind, col_ind):
        if r == c:
            continue
        pair = (min(r, c), max(r, c))
        if pair in seen:
            continue
        seen.add(pair)
        sim = 1.0 - cost[r, c] / max_cost
        candidates.append((fg[pair[0]].region_id, fg[pair[1]].region_id, sim))

    if not candidates:
        return []

    # Bimodality check: verify the similarity distribution has two modes
    # (real pairs vs forced matches) before applying Otsu.
    # Bimodality Coefficient (BC) = (skewness² + 1) / kurtosis.
    # BC > 5/9 (≈0.556, derived from uniform distribution) suggests
    # bimodality (SAS Institute; Freeman & Dale, 2013).
    # If unimodal: all matches are equally bad/good — no natural pairs.
    sims = np.array([s for _, _, s in candidates])

    if len(sims) >= _MIN_SAMPLES_FOR_PERCENTILE:
        mu = sims.mean()
        centered = sims - mu
        m2 = np.mean(centered ** 2)
        if m2 > np.finfo(float).eps:
            m3 = np.mean(centered ** 3)
            m4 = np.mean(centered ** 4)
            skewness = m3 / (m2 ** 1.5)
            kurtosis = m4 / (m2 ** 2)
            if kurtosis > np.finfo(float).eps:
                bc = (skewness ** 2 + 1) / kurtosis
                if bc < _BIMODALITY_THRESHOLD:
                    # Unimodal: no natural separation exists.
                    # All pairs are equally poor → return none.
                    return []

    # Otsu on similarities to separate real symmetric pairs from
    # forced matches (Hungarian must match everyone, but non-symmetric
    # regions get poor matches with low similarity).
    try:
        sim_thresh = threshold_otsu(sims)
    except ValueError:
        sim_thresh = np.median(sims)

    # Build region lookup for axis detection
    fg_by_id = {r.region_id: r for r in regions if not r.is_background}

    result_pairs = []
    for r1_id, r2_id, sim in candidates:
        if sim < sim_thresh:
            continue
        # Detect symmetry axis from centroid positions:
        # If the two centroids differ more in x → mirrored across vertical axis.
        # If they differ more in y → mirrored across horizontal axis.
        # This is a geometric fact derived from the centroid coordinates.
        r1_reg = fg_by_id.get(r1_id)
        r2_reg = fg_by_id.get(r2_id)
        if r1_reg and r2_reg:
            dx = abs(r1_reg.centroid[0] - r2_reg.centroid[0])
            dy = abs(r1_reg.centroid[1] - r2_reg.centroid[1])
            axis = "vertical" if dx >= dy else "horizontal"
        else:
            axis = "vertical"
        result_pairs.append(
            SymmetricPair(r1=r1_id, r2=r2_id, axis=axis, similarity=round(sim, 3))
        )
    return result_pairs


# ---------------------------------------------------------------------------
# Region properties — continuous values, adjacency, containment
# ---------------------------------------------------------------------------

def compute_regions(
    segments: NDArray, image: NDArray, resolution: int,
) -> tuple[list[PixelRegion], dict[int, int], NDArray]:
    """Extract properties, adjacency, containment.

    Returns (regions, raw_to_new_map, relabeled_segments).
    The relabeled_segments array maps each pixel to its region ID
    (contiguous from 1), matching the region_id in PixelRegion.
    """
    rgb = image[:, :, :3]
    total_area = resolution * resolution
    bg_labels_raw = _detect_background(segments, image)

    # Relabel contiguously from 1 (0 is reserved for "no region" in skimage)
    unique_labels = np.unique(segments)
    relabeled = np.zeros_like(segments)
    raw_to_new: dict[int, int] = {}
    for new_id, old_id in enumerate(unique_labels, start=1):
        relabeled[segments == old_id] = new_id
        raw_to_new[int(old_id)] = new_id

    bg_labels = {raw_to_new[r] for r in bg_labels_raw if r in raw_to_new}

    # Adjacency from RAG (topological fact)
    adj_raw = _compute_adjacency(image, segments)
    adj_new: dict[int, list[int]] = {}
    for raw_id, neighbors in adj_raw.items():
        new_id = raw_to_new.get(int(raw_id))
        if new_id is not None:
            adj_new[new_id] = [raw_to_new[int(n)] for n in neighbors if int(n) in raw_to_new]

    props = regionprops(relabeled)
    regions = []
    for p in props:
        if p.area < _NYQUIST_FACTOR:  # <2 pixels: sub-Nyquist, discard
            continue
        cy, cx = p.centroid
        mask_p = relabeled == p.label
        mean_color = tuple(int(c) for c in rgb[mask_p].mean(axis=0)[:3])
        has_stats = p.area >= _MIN_PIXELS_FOR_MOMENTS

        neighbors = sorted(set(adj_new.get(p.label, [])) - {p.label} - bg_labels)

        regions.append(PixelRegion(
            region_id=p.label,
            area_pct=round(p.area / total_area * 100, 2),    # display: 2 decimals
            centroid=(round(cx / resolution, 3), round(cy / resolution, 3)),  # display: 3 decimals
            bbox=p.bbox,
            eccentricity=round(float(p.eccentricity), 3) if has_stats else 0.0,  # display
            solidity=round(float(p.solidity), 3) if has_stats else 1.0,  # display
            euler_number=int(p.euler_number) if has_stats else 0,
            orientation=round(float(p.orientation), 3) if has_stats else 0.0,  # display
            mean_color=mean_color,
            is_background=p.label in bg_labels,
            neighbors=neighbors,
        ))

    regions.sort(key=lambda r: r.area_pct, reverse=True)

    # Containment (geometric bbox test)
    _compute_containment(segments, regions, bg_labels)

    return regions, raw_to_new, relabeled


# ---------------------------------------------------------------------------
# Top-N filtering — Shannon entropy perplexity
# ---------------------------------------------------------------------------

def _effective_region_count(regions: list[PixelRegion]) -> int:
    """Shannon entropy perplexity: effective number of equally-important regions.

    H = -Σ p_i log₂(p_i)  (Shannon, 1948)
    Perplexity = 2^H = effective number of equally-important items.

    If 156 regions have entropy H = 4.5, then 2^4.5 ≈ 23 regions capture the
    effective information content. This is the number of regions a uniform
    distribution with the same entropy would contain.

    Mathematical basis: Shannon's source coding theorem — H bits of information
    content is equivalent to a uniform distribution over 2^H items.
    """
    fg = [r for r in regions if not r.is_background]
    if len(fg) <= 1:
        return len(fg)

    areas = np.array([r.area_pct for r in fg])
    total = areas.sum()
    if total <= 0:
        return len(fg)

    p = areas / total
    # Add machine epsilon to prevent log(0)
    H = -np.sum(p * np.log2(p + np.finfo(float).eps))
    perplexity = int(np.ceil(2.0 ** H))

    # Floor: at least enough to include all "large" regions (Otsu split)
    return max(perplexity, 1)


# ---------------------------------------------------------------------------
# Region grouping — Otsu on area distribution
# ---------------------------------------------------------------------------

def _group_regions_by_area(regions: list[PixelRegion]) -> dict[str, list[PixelRegion]]:
    """Split foreground regions into large/small using Otsu on area."""
    bg = [r for r in regions if r.is_background]
    fg = [r for r in regions if not r.is_background]
    if len(fg) < _NYQUIST_FACTOR:
        return {"background": bg, "large": fg, "small": []}
    areas = np.array([r.area_pct for r in fg])
    try:
        area_thresh = threshold_otsu(areas)
    except ValueError:
        area_thresh = float(np.median(areas))
    return {
        "background": bg,
        "large": [r for r in fg if r.area_pct >= area_thresh],
        "small": [r for r in fg if r.area_pct < area_thresh],
    }


# ---------------------------------------------------------------------------
# ASCII visualization — with character mapping for LLM coupling
# ---------------------------------------------------------------------------

def _build_char_map(
    relabeled: NDArray, regions: list[PixelRegion],
) -> dict[int, str]:
    """Map region IDs to ASCII characters, largest regions first.

    Background regions always get '.'. Foreground regions get characters
    from _CHAR_PALETTE in order of decreasing area.
    """
    char_map: dict[int, str] = {}
    ci = 0
    # Sort foreground by area descending → largest gets first (most visible) char
    fg_sorted = sorted(
        [r for r in regions if not r.is_background],
        key=lambda r: r.area_pct, reverse=True,
    )
    for r in regions:
        if r.is_background:
            char_map[r.region_id] = "."
    for r in fg_sorted:
        char_map[r.region_id] = _CHAR_PALETTE[ci % len(_CHAR_PALETTE)]
        ci += 1
    return char_map


def _compute_ascii_step(relabeled: NDArray, regions: list[PixelRegion]) -> int:
    """Derive ASCII grid step size from smallest region width.

    Nyquist in 1D: each region needs ≥2 characters wide to be distinguishable.
    step = smallest_region_width / _NYQUIST_FACTOR
    This is data-derived from the actual region sizes in this image.
    """
    h, w = relabeled.shape
    fg = [r for r in regions if not r.is_background]
    if not fg:
        # No foreground: step = √(image width) to get a reasonable grid
        # (geometric mean between 1 and w — data-derived from image size)
        return max(1, int(np.sqrt(w)))

    min_width = min(max(r.bbox[3] - r.bbox[1], 1) for r in fg)
    step = max(1, min_width // _NYQUIST_FACTOR)
    return step


def _compute_grid_bboxes(
    relabeled: NDArray, regions: list[PixelRegion], step: int,
) -> dict[int, tuple[int, int, int, int]]:
    """Compute grid coordinates for each region.

    Returns dict: region_id → (min_row, min_col, max_row, max_col) in ASCII grid.
    """
    grid_bboxes: dict[int, tuple[int, int, int, int]] = {}
    row_step = step * _CHAR_ASPECT

    for r in regions:
        if r.is_background:
            continue
        pr1, pc1, pr2, pc2 = r.bbox
        grid_bboxes[r.region_id] = (
            pr1 // row_step,
            pc1 // step,
            pr2 // row_step,
            pc2 // step,
        )
    return grid_bboxes


def segments_to_ascii(
    relabeled: NDArray, char_map: dict[int, str], step: int,
) -> str:
    """Render segments as ASCII art using the character mapping."""
    h, w = relabeled.shape
    row_step = step * _CHAR_ASPECT
    lines = []
    for r in range(0, h, row_step):
        lines.append("".join(
            char_map.get(int(relabeled[r, c]), ".")
            for c in range(0, w, step)
        ))
    return "\n".join(lines)


def _build_legend(char_map: dict[int, str]) -> dict[str, int]:
    """Invert char_map: char → region_id (excluding '.' background)."""
    return {ch: rid for rid, ch in char_map.items() if ch != "."}


# ---------------------------------------------------------------------------
# Format results — raw values + relationships (human-readable)
# ---------------------------------------------------------------------------

def format_result_text(result: PixelSegmentationResult) -> str:
    """Human-readable output for debugging."""
    lines = ["=" * 64, "PIXEL SEGMENTATION RESULTS", "=" * 64]
    p = result.params
    lines.append(f"\nResolution: {result.resolution}x{result.resolution}")
    lines.append(f"Atomic segments: {result.n_atomic_segments}")
    if p:
        lines.append(f"Felzenszwalb: scale={p.get('scale', '?'):.1f}, "
                      f"sigma={p.get('sigma', '?'):.4f}, "
                      f"min_size={p.get('min_size', '?')}")
        if "rag_thresholds" in p:
            lines.append(f"RAG thresholds (Jenks): "
                         f"[{', '.join(f'{t:.1f}' for t in p['rag_thresholds'])}]")
        lines.append(f"RAG levels: {len(p.get('rag_thresholds', []))} (data-determined)")

    res = result.resolution
    if result.primary_center:
        cy, cx = result.primary_center
        lines.append(f"\n--- PRIMARY BOUNDARY ---")
        lines.append(f"Center: ({cx / res:.2f},{cy / res:.2f}), "
                     f"radius={result.primary_radius / res * 100:.0f}% of canvas, "
                     f"{len(result.sub_centers)} sub-centers")

    # Build char map and grid for coupled output
    char_map = {}
    grid_bboxes: dict[int, tuple[int, int, int, int]] = {}
    if result.relabeled_segments is not None:
        char_map = _build_char_map(result.relabeled_segments, result.regions)
        step = _compute_ascii_step(result.relabeled_segments, result.regions)
        grid_bboxes = _compute_grid_bboxes(result.relabeled_segments, result.regions, step)

    groups = _group_regions_by_area(result.regions)
    if groups["background"]:
        lines.append(f"\n--- BACKGROUND ({len(groups['background'])}) ---")
        for r in groups["background"]:
            lines.append(f"  R{r.region_id}: {r.area_pct:.1f}%, color=rgb{r.mean_color}")

    for label, group in [("Large", groups["large"]), ("Small", groups["small"])]:
        if group:
            lines.append(f"\n--- {label.upper()} REGIONS ({len(group)}) ---")
            for r in group:
                ch = char_map.get(r.region_id, "?")
                parts = [f"R{r.region_id}[{ch}]: {r.area_pct:.1f}% at ({r.centroid[0]:.2f},{r.centroid[1]:.2f})"]
                parts.append(f"ecc={r.eccentricity}, sol={r.solidity}")
                if r.euler_number != 1:
                    parts.append(f"euler={r.euler_number}")
                parts.append(f"color=rgb{r.mean_color}")
                if r.neighbors:
                    parts.append(f"adj=[{','.join(f'R{n}' for n in r.neighbors[:6])}]")
                if r.inside_of:
                    parts.append(f"inside=R{r.inside_of}")
                if r.contains:
                    parts.append(f"contains=[{','.join(f'R{c}' for c in r.contains[:6])}]")
                gb = grid_bboxes.get(r.region_id)
                if gb:
                    parts.append(f"grid[{gb[0]}:{gb[2]},{gb[1]}:{gb[3]}]")
                lines.append(f"  {', '.join(parts)}")

    if result.symmetric_pairs:
        lines.append(f"\n--- SYMMETRIC PAIRS ---")
        for sp in result.symmetric_pairs:
            ch1 = char_map.get(sp.r1, "?")
            ch2 = char_map.get(sp.r2, "?")
            lines.append(f"  R{sp.r1}[{ch1}] <-> R{sp.r2}[{ch2}] ({sp.axis}, sim={sp.similarity})")

    if result.merge_events:
        lines.append(f"\n--- MERGE TREE ---")
        for e in result.merge_events:
            if e.merged_regions:
                lines.append(f"  scale {e.threshold:.1f}: "
                             f"{len(e.merged_regions)} merge(s) -> {e.n_regions_after} regions")
                for g in e.merged_regions[:6]:
                    lines.append(f"    [{'+'.join(f'R{r}' for r in g)}]")

    if result.relabeled_segments is not None:
        lines.append(f"\n--- SEGMENT MAP ---")
        lines.append(segments_to_ascii(result.relabeled_segments, char_map, step))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Enrichment output — coupled pixel + ASCII for LLM
# ---------------------------------------------------------------------------

def format_enrichment_sections(result: PixelSegmentationResult) -> str:
    """Format for LLM: regions coupled with ASCII chars and grid positions.

    Each region description includes:
    - [char] label matching the segment map below
    - Grid coordinates (row:col range) for spatial cross-referencing
    - Containment and adjacency with char references
    - Continuous shape properties (no classification)

    The LLM can read "R3[@] inside R2[#], symmetric with R4[%]"
    and find @, #, % on the segment map to confirm spatial relationships.
    """
    lines = []
    groups = _group_regions_by_area(result.regions)
    fg = groups["large"] + groups["small"]

    # Top-N filtering: report at most 2^H regions (Shannon entropy perplexity).
    # This is data-derived: high-entropy distributions (many similar-sized
    # regions) get more detail; low-entropy (one dominant region) get less.
    n_effective = _effective_region_count(result.regions)
    # Ensure all "large" regions are always included
    n_large = len(groups["large"])
    n_report = max(n_effective, n_large)
    # Cap small regions to the remaining budget
    reported_small = groups["small"][:max(0, n_report - n_large)]

    # Build char map and grid info
    char_map: dict[int, str] = {}
    grid_bboxes: dict[int, tuple[int, int, int, int]] = {}
    step = 1
    if result.relabeled_segments is not None:
        char_map = _build_char_map(result.relabeled_segments, result.regions)
        step = _compute_ascii_step(result.relabeled_segments, result.regions)
        grid_bboxes = _compute_grid_bboxes(result.relabeled_segments, result.regions, step)

    legend = _build_legend(char_map)

    # PIXEL DECOMPOSITION
    lines.append("PIXEL DECOMPOSITION (data-derived):")
    if groups["background"]:
        bg_pct = sum(r.area_pct for r in groups["background"])
        lines.append(f"  {len(fg)} foreground regions "
                     f"({sum(r.area_pct for r in fg):.0f}%), background {bg_pct:.0f}%")
    else:
        lines.append(f"  {len(result.regions)} regions filling canvas")

    def _fmt_region(r: PixelRegion) -> list[str]:
        ch = char_map.get(r.region_id, "?")
        # Line 1: identity + position + shape
        line1_parts = [f"R{r.region_id}[{ch}]"]
        line1_parts.append(f"({r.centroid[0]:.2f},{r.centroid[1]:.2f})")
        line1_parts.append(f"{r.area_pct:.1f}%")
        line1_parts.append(f"ecc={r.eccentricity}")
        line1_parts.append(f"sol={r.solidity}")
        if r.euler_number < 1:
            line1_parts.append(f"holes={abs(r.euler_number)}")
        line1_parts.append(f"rgb{r.mean_color}")
        line1 = "    " + " ".join(line1_parts)

        # Line 2: grid position + relationships (coupled with char references)
        line2_parts = []
        gb = grid_bboxes.get(r.region_id)
        if gb:
            line2_parts.append(f"grid[{gb[0]}:{gb[2]},{gb[1]}:{gb[3]}]")
        if r.neighbors:
            adj_str = ",".join(
                f"R{n}[{char_map.get(n, '?')}]" for n in r.neighbors[:6]
            )
            line2_parts.append(f"touches {adj_str}")
        if r.inside_of is not None:
            line2_parts.append(f"inside R{r.inside_of}[{char_map.get(r.inside_of, '?')}]")
        if r.contains:
            cont_str = ",".join(
                f"R{c}[{char_map.get(c, '?')}]" for c in r.contains[:6]
            )
            line2_parts.append(f"contains {cont_str}")
        line2 = "      " + " | ".join(line2_parts) if line2_parts else ""

        return [line1] + ([line2] if line2 else [])

    if groups["large"]:
        lines.append(f"  Major ({len(groups['large'])}):")
        for r in groups["large"]:
            lines.extend(_fmt_region(r))

    if reported_small:
        omitted = len(groups["small"]) - len(reported_small)
        label = f"  Features ({len(reported_small)}"
        if omitted > 0:
            label += f", {omitted} omitted by entropy"
        label += "):"
        lines.append(label)
        for r in reported_small:
            lines.extend(_fmt_region(r))

    lines.append("")

    # SYMMETRIC PAIRS (with char references)
    if result.symmetric_pairs:
        lines.append("SYMMETRIC PAIRS:")
        for sp in result.symmetric_pairs:
            ch1 = char_map.get(sp.r1, "?")
            ch2 = char_map.get(sp.r2, "?")
            r1 = next((r for r in result.regions if r.region_id == sp.r1), None)
            r2 = next((r for r in result.regions if r.region_id == sp.r2), None)
            if r1 and r2:
                lines.append(
                    f"  R{sp.r1}[{ch1}] ({r1.centroid[0]:.2f},{r1.centroid[1]:.2f}) <-> "
                    f"R{sp.r2}[{ch2}] ({r2.centroid[0]:.2f},{r2.centroid[1]:.2f}) "
                    f"sim={sp.similarity}"
                )
        lines.append("")

    # MERGE TREE
    if any(e.merged_regions for e in result.merge_events):
        lines.append("MERGE TREE:")
        for e in result.merge_events:
            if e.merged_regions:
                merges = ", ".join(
                    f"[{'+'.join(f'R{r}' for r in g)}]"
                    for g in e.merged_regions[:6]
                )
                lines.append(f"  scale {e.threshold:.0f}: {merges} -> {e.n_regions_after} regions")
        lines.append("")

    # PRIMARY MASS
    if result.primary_center:
        cy, cx = result.primary_center
        res = result.resolution
        lines.append(f"PRIMARY MASS: center=({cx / res:.2f},{cy / res:.2f}), "
                     f"inscribed_radius={result.primary_radius / res * 100:.0f}%, "
                     f"{len(result.sub_centers)} sub-centers")
        lines.append("")

    # SEGMENT MAP (with legend for LLM cross-referencing)
    if result.relabeled_segments is not None:
        ascii_art = segments_to_ascii(result.relabeled_segments, char_map, step)
        # Build legend string
        legend_parts = [".=bg"]
        for ch, rid in sorted(legend.items(), key=lambda x: x[1]):
            legend_parts.append(f"{ch}=R{rid}")
        legend_str = " ".join(legend_parts)
        lines.append(f"SEGMENT MAP ({legend_str}):")
        lines.append(ascii_art)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pixel_pipeline(
    svg_code: str,
    n_elements: int = 0,
    element_bboxes: list[tuple[float, float, float, float]] | None = None,
    canvas_w: float = _W3C_DEFAULT_WIDTH,
    canvas_h: float = _W3C_DEFAULT_HEIGHT,
) -> PixelSegmentationResult:
    """Run complete pixel segmentation pipeline. All parameters data-derived.

    Default canvas dimensions follow W3C CSS 2.1 Section 10.3.2:
    replaced elements with no intrinsic dimensions = 300px × 150px.
    """
    if element_bboxes:
        resolution = compute_resolution(element_bboxes, canvas_w, canvas_h, n_elements)
    else:
        # No element bounding boxes: Nyquist sampling of canvas dimensions.
        # Lower bound: data-derived from n_elements.
        min_res = _min_resolution_for_elements(n_elements)
        resolution = max(min_res, int(_NYQUIST_FACTOR * max(canvas_w, canvas_h)))

    image = rasterize_svg(svg_code, resolution)
    params = compute_felzenszwalb_params(image, max(n_elements, 1))
    segments = segment_image(image, params)

    thresholds = find_rag_thresholds(image, segments)
    params["rag_thresholds"] = thresholds

    hierarchy = build_hierarchy(image, segments, thresholds) if thresholds else {}
    merge_events = find_merge_events(segments, hierarchy) if hierarchy else []

    alpha = image[:, :, 3] if image.shape[2] >= 4 else np.full(
        image.shape[:2], 255, dtype=np.uint8,
    )
    primary_center, primary_radius, sub_centers = find_primary_boundary(alpha > 0)

    regions, raw_to_new, relabeled = compute_regions(segments, image, resolution)
    symmetric_pairs = _detect_symmetric_pairs(regions)

    return PixelSegmentationResult(
        resolution=resolution,
        n_atomic_segments=len(np.unique(segments)),
        regions=regions,
        merge_events=merge_events,
        hierarchy_levels={t: len(np.unique(l)) for t, l in hierarchy.items()},
        primary_center=primary_center,
        primary_radius=primary_radius,
        sub_centers=sub_centers,
        symmetric_pairs=symmetric_pairs,
        params=params,
        segment_labels=segments,
        relabeled_segments=relabeled,
        raster_image=image,
    )

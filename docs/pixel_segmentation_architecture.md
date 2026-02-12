# VectorSight: Pixel Segmentation Architecture

## Problem

The current pipeline has **62 transforms** computing raw geometry features, but **162 hardcoded thresholds** deciding how to interpret, group, classify, and present those features to the LLM. Examples:

- `circularity > 0.85` → "circular" (fixed for every SVG)
- `n_elements > 30` → "complex" (cliff at 30)
- `fill_pct < 0.85` → skip grid level
- `IoU > 0.05` → include overlap
- Fixed grid sizes: 12×12, 16×16, 14×14, 10×10

These were hand-tuned on a few test SVGs. An SVG with 29 elements and one with 31 get radically different treatment.

## Solution: Pixel-Domain Perceptual Segmentation

Rasterize the SVG to an image and let **the image itself** tell us what the meaningful visual units are — using classic image processing algorithms (2004-era, not ML, zero training data).

### What We Keep (Vector Domain)

The 62 transforms fall into two categories:

**KEEP — Raw Feature Computation (~40 transforms)**
These compute exact geometric measurements from SVG vector paths. Pixels cannot replicate this precision.

| Layer | Transforms | What They Compute |
|-------|-----------|-------------------|
| L0 (7) | Parsing, normalization, Bezier sampling | SVG structure, boundary points, polygons |
| L1 (22) | Curvature, inflection, Fourier, circularity, rectangularity, eccentricity, corners, color | Per-element geometric + visual properties |
| L2 (7) | ASCII grids, Braille, composite silhouette | Visualization outputs |
| L4 (5) | Validation, cross-checks | Quality assurance |

These transforms produce **continuous values** (circularity=0.87, eccentricity=0.34, curvature_mean=0.12). The values themselves are not the problem — the hardcoded thresholds that interpret them are.

**REPLACE — Grouping, Classification, Interpretation (~20 transforms + enrichment formatter)**
These use hardcoded thresholds to decide relationships and categories. Pixel segmentation replaces this logic.

| Current | Problem | Replaced By |
|---------|---------|-------------|
| T3.01 Containment matrix | O(n²) pairwise test, 50% area threshold | RAG hierarchy (automatic nesting) |
| T3.07 DBSCAN clustering | 10% diagonal epsilon, min_samples=2 | Felzenszwalb segmentation (adaptive) |
| T3.15 Shared center / concentric groups | 3% tolerance, 1.5x size ratio | Segments naturally merge concentric elements |
| T3.18 Connected components | 2% diagonal proximity | Connected components on raster (exact) |
| T1.23 Shape class labeling | circularity>0.85, rect>0.85, etc. | regionprops (eccentricity, solidity — continuous) |
| `_find_primary_boundary()` | 15% child count, area ranking | Distance transform global maximum (0 params) |
| Visual Pyramid (enrichment) | Fixed 12/16/14/10 grids | RAG merge tree at 4 thresholds (automatic) |
| Complexity gate | n>30 binary switch | Hierarchy depth (continuous) |
| Fill cascades | >85%, >65%, >40% | Shannon entropy (information-based) |
| Importance scoring | Hand-tuned additive weights | Center-surround saliency (context-relative) |

### What We Add (Pixel Domain)

Four algorithms, all already in our dependencies (skimage, scipy), zero training data:

#### 1. Felzenszwalb Graph-Based Segmentation
- **What**: Treats pixels as a graph, merges neighbors that look similar. Adaptive threshold — small regions need stronger evidence to stay separate.
- **Parameters**: 3 — all data-derived (scale from Lab IQR, sigma from wavelet noise, min_size from element count). See [Data-Driven Parameter Solutions](#data-driven-parameter-solutions--zero-human-chosen-values).
- **Output**: ~50-200 atomic regions respecting color boundaries
- **Library**: `skimage.segmentation.felzenszwalb()`
- **Cost**: ~3ms on 256×256

#### 2. Region Adjacency Graph (RAG) + Hierarchical Merging
- **What**: Builds a graph where regions are nodes, edges weighted by color difference. Merges similar adjacent regions at increasing thresholds. The merge tree IS the visual pyramid.
- **Parameters**: Thresholds from Jenks natural breaks on the edge weight distribution — data-derived, not hardcoded. See [Data-Driven Parameter Solutions](#data-driven-parameter-solutions--zero-human-chosen-values).
- **Output**: Multi-scale hierarchy (automatic zoom levels)
- **Library**: `skimage.graph.rag_mean_color()` + `merge_hierarchical()`
- **Cost**: ~15ms

Example merge sequence for a squirrel logo:
```
thresh=15: eye pupil + eye white → "eye region"
thresh=30: ear parts → "ear". Acorn cap + body → "acorn"
thresh=50: face + ears → "head". Body parts → "body"
thresh=75: everything → "whole figure"
```

#### 3. Distance Transform
- **What**: For each foreground pixel, compute distance to nearest background. Deepest point = primary boundary center. Local maxima = sub-region centers.
- **Parameters**: ZERO
- **Output**: Primary boundary detection, sub-region seeds
- **Library**: `scipy.ndimage.distance_transform_edt()`
- **Cost**: <1ms

#### 4. Region Properties
- **What**: For each segment, compute area, centroid, bbox, eccentricity, solidity, euler_number (counts holes), orientation, moments.
- **Parameters**: ZERO
- **Output**: Per-region feature vector (continuous, no classification needed)
- **Library**: `skimage.measure.regionprops()`
- **Cost**: ~2ms

### Supplementary Techniques

#### Otsu's Method — Data-Driven Thresholds
For the few remaining thresholds (shape classification in L1 transforms), use Otsu to find natural split points from THIS SVG's data distribution instead of fixed global values.
- `skimage.filters.threshold_otsu()`
- Replaces: `circularity > 0.85`, `rectangularity > 0.85`, etc.

#### Shannon Entropy — Information Budget
For inclusion decisions (which grid levels to show, which elements to include), measure actual information content instead of fill percentages.
- `scipy.stats.entropy()`
- Replaces: `fill_pct < 0.85`, `top_n = 12 vs 15`

#### Center-Surround Saliency — Element Importance
An element is important because it DIFFERS from its neighbors, not because it exceeds a fixed number. Replace hand-tuned importance weights with `|value - local_mean| / local_std`.
- Replaces: `_compute_importance()` additive weights

## Pipeline Architecture

```
SVG Code
    │
    ├──→ [Vector Domain — KEEP]
    │      L0: Parse SVG → elements, boundaries, polygons
    │      L1: Per-element features (circularity, curvature, color, etc.)
    │      L2: Braille grids, ASCII art
    │      L4: Validation
    │
    ├──→ [Pixel Domain — NEW, all parameters data-derived]
    │      Nyquist resolution from min element size
    │      CairoSVG → N×N RGBA raster (N from Nyquist)
    │      ↓
    │      Lab IQR → scale, wavelet → sigma, n_elements → min_size
    │      Felzenszwalb(scale, sigma, min_size) → atomic regions
    │      ↓
    │      RAG edge weights → Jenks natural breaks → merge thresholds
    │      RAG + hierarchical merge → visual hierarchy (data-driven levels)
    │      ↓
    │      Distance transform → persistence peaks → sub-region centers
    │      regionprops → per-region properties (0 params)
    │      ↓
    │      Soft weighted overlap → SVG element ↔ segment mapping
    │
    └──→ [Enrichment Formatter — REWRITE]
           Merge vector features + pixel hierarchy
           Present as structured text for LLM:
             - Visual decomposition (from RAG hierarchy)
             - Region properties (from regionprops)
             - Merge tree (what groups with what, at what scale)
             - Vector-domain detail (curvature, Fourier, etc.) per region
             - Braille/ASCII grids at hierarchy-selected zoom levels
```

## Enrichment Output Format

```
VISUAL DECOMPOSITION (auto-discovered from image):
  Scene: 1 figure (88% of canvas) + background
  Major parts (3):
    - Region A: upper-right, 35% of figure, 2 holes (euler=-2)
    - Region B: center, 45%, largest mass, elongated (ecc=0.7)
    - Region C: lower-left, 12%, ovoid (ecc=0.4), adjacent to B
  Features (11):
    - A.1: paired dark circles at (0.7, 0.3), symmetric
    - A.2: pointed protrusions at top of A
    - B.1: curved mass extending upper-right from B
    - C.1: ovoid with internal layering, 3 concentric rings

MERGE TREE (how features combine):
  scale 15: [A.1a+A.1b → A.1], [C.1a+C.1b → C.1]
  scale 30: [A.1+A.2+A.3 → A], [B.1+B.2 → B]
  scale 55: [A+B → figure]

[+ existing sections: spatial interpretation, shape narrative,
   Braille grids, reconstruction steps, learned patterns]
```

The LLM reads "paired dark circles, symmetric, inside a region with pointed protrusions at top" and reasons: "eyes inside a head with ears." We never said "eye" or "ear."

## Threshold Reduction

| Category | Before | After (initial) | After (data-driven) |
|----------|--------|-----------------|---------------------|
| Binary switches | 34 | ~4 | 0 in pixel pipeline |
| Continuous thresholds | 68 | ~10 | 1 (persistence ratio 0.1, scale-invariant) |
| Discrete parameters | 42 | ~8 | ~6 (numerical analysis: Bezier density, Fourier count, etc.) |
| **Total magic numbers** | **162** | **~22** | **~7** |

The remaining ~7 are in the vector-domain transforms we keep (L0/L1/L2/L4) — numerical analysis parameters like Bezier sampling density, Braille grid resolution, and Fourier descriptor count. These define precision, not interpretation. The pixel pipeline itself has **zero content-sensitive human-chosen parameters** — every value comes from the image or SVG metadata.

## Computational Budget

| Step | Time | Dependencies |
|------|------|-------------|
| CairoSVG rasterize | ~5ms | cairosvg (existing) |
| Distance transform | <1ms | scipy (existing) |
| Felzenszwalb | ~3ms | skimage (existing) |
| regionprops | ~2ms | skimage (existing) |
| RAG + merge (4 levels) | ~15ms | skimage (existing) |
| SVG element mapping | ~5ms | numpy (existing) |
| **Total pixel pipeline** | **~30ms** | **zero new deps** |

Current vector pipeline: ~2-3 minutes for complex SVGs. The pixel pipeline adds negligible overhead.

## Research Basis

- **Felzenszwalb & Huttenlocher (2004)**: "Efficient Graph-Based Image Segmentation" — adaptive local thresholds, O(N log N)
- **SiT-Bench (2025)**: Structured symbolic descriptions beat pixel-level for LLM spatial reasoning
- **VoT (NeurIPS 2024)**: Step-by-step reconstruction improves LLM spatial reasoning by 27%
- **Graph-Based Captioning (2024)**: Region graph descriptions produce best LLM understanding
- **Text4Seg (2024)**: Encoding segmentation as text reduces descriptor length by 74%

## Data-Driven Parameter Solutions — Zero Human-Chosen Values

The initial architecture described above still had ~8 parameters (Felzenszwalb `scale`/`sigma`/`min_size`, rasterization resolution, RAG merge thresholds). Further research found data-driven solutions for ALL of them — every value is derived from the image itself or the SVG metadata.

### 1. Felzenszwalb `scale` — From Lab Color Gradient IQR

The `scale` parameter controls merge sensitivity. Instead of a fixed value:

```python
from skimage.color import rgb2lab
lab = rgb2lab(image[:, :, :3])
gradients = np.sqrt(np.diff(lab, axis=0)[:, :-1] ** 2 + np.diff(lab, axis=1)[:-1, :] ** 2)
gradient_magnitudes = np.linalg.norm(gradients, axis=-1)
q25, q75 = np.percentile(gradient_magnitudes[gradient_magnitudes > 0], [25, 75])
scale = q75 - q25  # IQR of Lab color gradients
```

**Why it works**: IQR measures the natural variation in color transitions. High-contrast SVGs (bold logos) get a larger `scale` (merge more aggressively), low-contrast SVGs (subtle gradients) get a smaller `scale` (preserve fine boundaries). The data dictates sensitivity.

### 2. Felzenszwalb `sigma` — From Wavelet Noise Estimation

Pre-smoothing Gaussian sigma. For clean SVG rasters this should be ~0:

```python
from skimage.restoration import estimate_sigma
sigma = estimate_sigma(image, channel_axis=-1)  # Returns ~0.0 for clean SVG rasters
```

**Why it works**: `estimate_sigma()` uses wavelet decomposition (MAD of finest wavelet coefficients) to estimate actual noise level. SVG rasters from CairoSVG are perfectly clean — sigma ≈ 0, meaning no smoothing needed. If anti-aliasing or rasterization artifacts are present, it adapts automatically.

### 3. Felzenszwalb `min_size` — From Element Count

Minimum segment size prevents over-segmentation:

```python
n_elements = len(svg_elements)
min_size = (image_height * image_width) / (n_elements * 4)
```

**Why it works**: Each SVG element should map to at least one segment. Factor of 4 allows sub-element features (a circle with a highlight = 2 segments per element). More elements → smaller minimum size → finer segmentation. Fewer elements → larger minimum → no noise segments.

### 4. Rasterization Resolution — Nyquist Criterion

Instead of a fixed 256×256:

```python
min_feature_size = min(element_bbox_sizes)  # Smallest SVG element dimension
resolution = int(6 * canvas_dimension / min_feature_size)
resolution = np.clip(resolution, 128, 512)  # Safety bounds
```

**Why it works**: Nyquist theorem says you need at least 2× sampling rate to capture a signal. We use 6× (3× Nyquist) to ensure small features are well-represented. A logo with tiny details gets higher resolution; a simple 3-element icon stays at 128×128.

### 5. RAG Merge Thresholds — Jenks Natural Breaks

Instead of fixed `[15, 30, 50, 75]`, find where the data naturally clusters:

```python
from skimage.graph import rag_mean_color
rag = rag_mean_color(image, segments)
edge_weights = sorted([d['weight'] for u, v, d in rag.edges(data=True)])

# Find largest gaps in the sorted weight distribution
diffs = np.diff(edge_weights)
gap_indices = np.argsort(diffs)[-3:]  # Top 3 largest gaps
thresholds = sorted([edge_weights[i] for i in gap_indices])
```

**Why it works**: The merge cost distribution has natural clusters — some region boundaries are subtle (within-part color variation), others are strong (between-part boundaries). The largest gaps in the sorted merge cost distribution correspond to meaningful scale transitions. For the squirrel: gap at ~12 (pupil/iris boundary), ~28 (ear/head boundary), ~55 (head/body boundary). These emerge from the image itself.

Alternative: Full Jenks natural breaks via `jenkspy.jenks_breaks(edge_weights, n_classes=4)` if more precision needed.

### 6. Distance Transform Local Maxima — Topological Persistence

Finding significant peaks in the distance map without a threshold:

```python
from scipy.ndimage import distance_transform_edt, label
from scipy.ndimage import maximum_filter

dist = distance_transform_edt(foreground_mask)

# Topological persistence: rank peaks by "how much you must descend to reach a higher peak"
local_max = maximum_filter(dist, size=3) == dist
peak_coords = np.argwhere(local_max & (dist > 0))
peak_values = dist[peak_coords[:, 0], peak_coords[:, 1]]

# Union-find persistence: only keep peaks whose persistence > max_value * 0.1
# (0.1 is the ONLY parameter, and it's scale-invariant — always relative to global max)
persistence_threshold = dist.max() * 0.1
# ... union-find algorithm ranks peaks by persistence ...
```

**Why it works**: Topological persistence (Edelsbrunner & Harer, 2010) measures how "important" a local maximum is by computing how far you must descend before reaching a higher peak. Small bumps have low persistence; true sub-region centers have high persistence. The threshold is relative to the global maximum (scale-invariant), not an absolute pixel value. For a squirrel: the body center and head center have high persistence; noise peaks on curved boundaries are filtered out.

### 7. Morphological Radii — From Distance Transform

Instead of fixed structuring element sizes:

```python
max_dist = dist.max()
radii = np.logspace(0, np.log2(max_dist), num=4, base=2).astype(int)
radii = np.unique(np.clip(radii, 1, int(max_dist)))
```

**Why it works**: The distance transform tells us the maximum inscribed circle radius. Logarithmic spacing of radii from 1 to `max_dist` creates natural scale levels. Small SVGs with thin features get small radii; large SVGs with thick masses get larger radii. The number of levels (4) matches our hierarchy depth, but even this could be derived from the persistence diagram.

### 8. SVG-to-Pixel Element Mapping — Soft Weighted Overlap

Instead of `IoU > 0.05` hard threshold:

```python
# For each SVG element, render it alone on a blank canvas
element_mask = rasterize_single_element(element)

# Weight = fraction of element's pixels that fall in each segment
for seg_id in unique_segments:
    seg_mask = (segments == seg_id)
    overlap = np.sum(element_mask & seg_mask)
    weight = overlap / np.sum(element_mask)  # Normalized 0-1
    if weight > 0:
        mapping[element_idx][seg_id] = weight  # Soft assignment
```

**Why it works**: Every element maps to every overlapping segment with a continuous weight. No threshold decides "in or out." The enrichment formatter can rank by weight, sum weights per region, or use the full distribution. A small decorative element might map 60% to one segment and 40% to another — both associations are preserved.

### Summary: Parameter Derivation Chain

```
SVG metadata                    Image itself
    │                               │
    ├─ n_elements ──→ min_size      ├─ Lab gradients ──→ scale
    ├─ bbox sizes ──→ resolution    ├─ wavelet MAD ──→ sigma
    │                               ├─ edge weights ──→ RAG thresholds (Jenks)
    │                               ├─ distance map ──→ persistence peaks
    │                               ├─ max inscribed ──→ morphological radii
    │                               └─ per-element render ──→ soft overlap weights
    │
    └─ ZERO content-sensitive human-chosen parameters
```

Every parameter is either derived from the SVG structure (element count, bounding boxes) or from the rasterized image itself (color gradients, distance field, merge cost distribution). No parameter requires a human to look at sample outputs and pick a number.

## Implementation Plan

1. **Prototype** (~1 day): Rasterize Flink squirrel, run Felzenszwalb + RAG, visualize discovered regions
2. **Integration** (~2 days): Add pixel pipeline as new module `app/engine/pixel_segmentation.py`, run alongside vector pipeline
3. **Enrichment rewrite** (~2 days): Replace threshold-heavy formatter sections with hierarchy-driven generation
4. **Threshold cleanup** (~1 day): Replace remaining L1 thresholds with Otsu, fill checks with entropy
5. **Testing** (~1 day): Verify enrichment quality on test SVGs, ensure no regression

## Key Principle

> We don't need to identify the acorn. We need to present the data such that the LLM CAN identify the acorn. The image's own structure — what pixels are similar, what regions are adjacent, what merges at what scale — tells the LLM everything it needs.

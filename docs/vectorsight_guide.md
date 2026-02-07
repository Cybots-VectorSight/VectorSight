# VectorSight: Complete SVG Spatial Analysis Guide

## What This Document Is

You are an LLM receiving raw SVG path data. Your job is to:
1. **Analyze** — understand what shapes are depicted without rendering the SVG
2. **Reason** — determine the designer's intent (what they meant, not just what the coordinates say)
3. **Reconstruct** — output a semantically meaningful SVG (clean groups, transforms, CSS, IDs)
4. **Create & Edit** — place new elements precisely using spatial intent descriptions (the engine computes coordinates — you never do arithmetic)

This document is your complete instruction set. It contains:
- **PART 1:** How to think (reasoning protocol, critical thinking rules, resonance between steps)
- **PART 2:** What to compute (transformation catalog, 55 transforms across 5 layers)
- **PART 3:** How to output (enrichment summary format, semantic SVG reconstruction, representation decisions)
- **PART 4:** How to create and edit (spatial grammar, placement protocol, validation loop)
- **PART 5:** What goes wrong (failure taxonomy from 5 real SVGs, with fixes)
- **PART 6:** Execution checklists and pipeline architecture
- **PART 7:** Assumptions and open questions
- **PART 8:** Pipeline summary and implementation libraries
- **PART 9:** Self-improvement protocol (how this guide evolves from failures)

**Validated against:**
- Airbnb Bélo logo — organic filled shape with hole → SUCCEEDED
- Auth0 shield+star — composite filled shape with negative space → FAILED, then diagnosed
- Weather "partly cloudy" icon — stroke-based multi-element → SUCCEEDED
- Hugging Face logo — multi-element filled icon (6 paths) → SUCCEEDED (geometry alone)
- Apache Flink logo — single path, 4 sub-paths → SUCCEEDED (geometry + colour needed for brand)

### Design Principle: Give Data, Not Tools

The LLM doesn't need tools to call. It needs **better data**. Every transform below is pre-computed by the geometry engine and delivered as enrichment alongside the raw SVG. The LLM reads a structured summary in one pass and spends attention on reasoning, not measurement.

Don't give the LLM a tape measure and tell it to measure every room. Give it the floor plan.

### Design Principle: Information Density Spectrum

Inspired by VisionCortex's Impression system, which treats image simplification as "controlled information reduction" — removing shapes from least to most important while reducing fractal complexity within each shape. The same principle governs our transforms.

Every SVG exists on a spectrum from raw coordinates (maximum information, minimum meaning) to a semantic label like "smiley face" (minimum information, maximum meaning). Our pipeline is a controlled walk down this spectrum:

```
RAW COORDINATES          "M12,8 A2,2 0 1,1 12,4 C..."
    │                    Maximum info, zero meaning to LLM
    ▼
LAYER 0 (Parse)          Absolute coords, sub-paths extracted
    │                    Same info, cleaner format
    ▼
LAYER 1 (Features)       "circular, CV=0.09, area=314"
    │                    Less info, more meaning
    ▼
LAYER 2 (Visualization)  ASCII grid, width profile
    │                    Spatial info preserved in LLM-readable form
    ▼
LAYER 3 (Relationships)  "E2 inside E1, symmetric pair"
    │                    Structural meaning emerges
    ▼
ENRICHMENT SUMMARY       1,200 tokens of structured facts
    │                    Right level for LLM reasoning
    ▼
SEMANTIC LABEL           "smiley face"
                         Maximum meaning, minimum info
```

Each layer discards information the LLM can't use and adds meaning the LLM can. The key insight: **the LLM doesn't need maximum information — it needs the RIGHT information at the RIGHT abstraction level.** Our transforms are not random features. They are controlled steps down the information density spectrum, each producing a representation more useful to the LLM than the one above it.

This mirrors how VisionCortex's image tree walks from pixels (raw) → clusters (grouped) → hierarchical tree (structured) → stacked vector paths (semantic). Same architecture, different modality: they process pixels for algorithms, we process vector geometry for LLMs.

### Why Simple SVGs Work and Complex Ones Don't

Icons (1–15 paths, geometric primitives) work because the LLM can hold ~10 shapes in working memory. Complex SVGs (20–100+ paths, dense Bézier curves) break in five specific ways:

| Failure Mode | What Happens | Example |
|---|---|---|
| **Bézier blindness** | LLM sees control points but can't trace actual curve shape | Fox tail: 12 cubic Béziers — LLM knows endpoints but not the silhouette |
| **No spatial index** | With 60 elements, "is E34 near E12?" requires O(n²) attention | Animal with separate paths per fur tuft |
| **No shape classification** | LLM must manually deduce a 200-char path is "roughly circular" | Eye pupil encoded as 8 cubic Béziers |
| **No topology** | "Is E5 inside E2?" requires mental point-in-polygon testing | Spots on a ladybug — inside the wing or floating? |
| **Symmetry at scale** | LLM can eyeball bilateral symmetry with 6 elements, not 40 | Butterfly wings with 20 mirrored path pairs |

Observed accuracy degradation:

| Complexity | Paths | Accuracy | Bottleneck |
|---|---|---|---|
| Simple (geometric) | 1–5 | 9–10/10 | None |
| Simple (organic) | 1–10 | 9–10/10 | None |
| Medium | 10–30 | 7–8/10 | Pairwise relationships get fuzzy |
| Complex | 30–60 | 5–6/10 | Can't hold all relationships in attention |
| Illustrated | 60–100+ | 3–4/10 | Total overload |

The enrichment transforms in Part 2 specifically target the 30+ element range by pre-computing containment, clustering, symmetry, and shape classification — shifting measurement cost from LLM attention to geometry engine compute.

---

## PART 1: REASONING PROTOCOL

### 1.1 The Core Principle: PROVE, THEN INTERPRET

Your thinking has two modes. Keep them separate.

**GEOMETRIC PROOF** = deterministic, falsifiable, trustworthy
  "Centroid distance CV = 0.090" — fact. Reproducible.
  "5 sub-paths, each 1 line segment, length within 0.002 of 1.0" — fact.
  "Turning function total = 227.6°" — fact.

**SEMANTIC INTERPRETATION** = reasoning, defeasible, requires judgment
  "It's a sun" — interpretation built on facts.
  "The designer intended 45° spacing" — educated guess.
  "The missing rays mean occlusion" — inference from absence.

**Rule: exhaust geometric facts FIRST. Interpret SECOND. Never skip a measurement because you think you already know the answer.**

If you find yourself writing "this looks like a..." before you've computed turning function, centroid distance, and rendered the grid — STOP. You're interpreting before proving.

### 1.2 Step 0: Parse the SVG (Layer 0)

Expand all shorthand commands, resolve relative→absolute coordinates, extract sub-paths. This is mechanical — see Layer 0 transforms in Part 2.

After parsing, you have: a list of sub-paths, each being an ordered list of absolute-coordinate line/curve segments.

**CHECK THE FILL ATTRIBUTE FIRST:**
- `fill="[color]"` or no fill attribute → Filled shape. Use standard pipeline (ray casting, area metrics, width profile).
- `fill="none"` with `stroke="[color]"` → **STROKE-BASED icon.** The visual is the stroke line itself, not filled regions. Use turning function, centroid distance, directional coverage instead of area-based metrics. Render strokes as thick pixels (stroke-width) on the ASCII grid.

This single check determines which 60% of your transforms are relevant. Getting it wrong wastes everything downstream.

### 1.3 Step 1: Structural Triage

Before computing any features, answer these questions from the parsed data alone. This is FREE — costs almost nothing — and determines the entire strategy.

**Q0: Stroke or fill?**
- `fill=[color]` → filled shape pipeline (ray casting, area, width profile)
- `fill=none, stroke=[color]` → stroke-based pipeline (turning function, centroid distance, directional coverage)
- Both fill and stroke → treat as filled (stroke is secondary)

**Q1: How many sub-paths?**
- 1 sub-path → single solid shape (go to Step 2)
- 2 sub-paths with opposite winding → shape with hole/cutout (like Bélo)
- 2+ sub-paths with SAME winding → **COMPOSITE SHAPE — sub-paths are PIECES of a larger design. CRITICAL: analyze both what the pieces form AND what the gaps between them form.**
- Many separate `<path>` elements → multi-element icon (faces, scenes, etc.)

**Q2: Do sub-paths OVERLAP, TILE, or SEPARATE?**
- Overlapping bboxes + opposite winding = hole (even-odd fill rule)
- Adjacent/touching bboxes + same winding = **tiling composite** (Auth0 pattern)
- Well-separated bboxes = independent shapes

**Q3: What segment types?**
- All lines → polygon. Compute vertex angles, count corners.
- Mixed lines + curves → organic/designed shape. Full analysis needed.
- All curves → smooth organic form. Curvature profile is primary.

**Q4: Size clustering (for multi-sub-path SVGs)?**
- Cluster sub-paths by area into 3 tiers using natural breaks (Jenks) or log-scale k-means
- **LARGE** tier = structural outlines (body, head, shield)
- **MEDIUM** tier = features (eyes, ears, nose, spots)
- **SMALL** tier = details (highlights, whiskers, dots, micro-decorations)
- This immediately separates "the main shapes" from "the decorations"
- With 60+ elements, this is the first way to avoid drowning in detail

### 1.4 Step 2: Cheap Discriminators (run on EVERY curved sub-path)

These take <1ms each and answer fundamental structural questions. **Never skip these.**

| Transform | What it answers | Cost |
|---|---|---|
| Turning function (T1.3) | Open arc? Closed loop? How much rotation? | Trivial |
| Centroid distance CV (T1.4) | Circular? Elliptical? Complex? | Trivial |
| Segment count + types | How many lines vs curves? | Free (from parse) |
| Bbox aspect ratio | Tall? Wide? Square? | Free (from parse) |
| Closed? (start ≈ end) | Does it form a closed region? | Free (from parse) |

A turning function of 360° instantly tells you "closed loop." A centroid distance CV of 0.09 instantly tells you "circular." You don't need curvature profiles or Fourier analysis to know these things.

### 1.5 Step 3: Pattern Detection (relationships between sub-paths)

This is where most failures happen. The question: "are these N sub-paths N separate objects, or parts of ONE object?"

**Tests to run (in order):**
1. **Shared center test** — do multiple sub-paths share a centroid / radial center?
2. **Similarity test** — are any sub-paths congruent? Use Hu moment invariants (7 numbers, rotation/scale/translation invariant) with log-transform and DBSCAN (eps=0.5). Two shapes with similar Hu moments look alike regardless of position or size.
3. **Tiling test** — do adjacent sub-paths share edges / form a continuous fill?
4. **Nesting test** — is one sub-path inside another? Use **winding number algorithm**: for B's centroid, compute winding number w.r.t. A's boundary. Non-zero → B is inside A. For robustness, test 5 points (centroid + 4 bbox midpoints) and majority-vote.
5. **Angular spacing test** — are similar sub-paths arranged at regular intervals?
6. **Size clustering** — can sub-paths be grouped by area tier? (LARGE=structural, MEDIUM=features, SMALL=details)

Weather icon example: 5 sub-paths all have length≈1.0, all are single line segments, all at ~45° angular spacing from a common center, all at equal radial distance. That's a ROTATION PATTERN — detectable from geometry alone before any semantic reasoning.

Auth0 example: 3 sub-paths with adjacent bboxes, same winding direction, sharing edges. That's TILING — they form one composite shape, not three separate shapes.

### 1.6 Step 4: Dual-Space Analysis (THE CRITICAL STEP)

**ALWAYS analyze BOTH filled and empty regions.**

This is not optional. This is the step that catches shield+star, yin-yang, letter cutouts, logo negative space, stencil designs, and any shape where meaning lives in the gaps.

```
┌─────────────────────────────────────────┐
│         POSITIVE SPACE                  │
│  "What shape do the FILLED areas form?" │
│                                         │
│  → Compute ASCII grid of fills          │
│  → Compute composite silhouette         │
│  → Run width profile on fills           │
│  → Describe what you see                │
├─────────────────────────────────────────┤
│         NEGATIVE SPACE                  │
│  "What shape do the GAPS form?"         │
│                                         │
│  → Invert the grid (swap █ and ·)       │
│  → Count connected empty regions        │
│  → Run width profile on gaps            │
│  → Describe what you see                │
├─────────────────────────────────────────┤
│         FIGURE-GROUND DECISION          │
│  "Which one carries the meaning?"       │
│                                         │
│  → Often BOTH do (shield + star)        │
│  → The one with higher symmetry and     │
│    geometric regularity is usually      │
│    the intentional design element       │
│  → Report both. Let context decide.     │
└─────────────────────────────────────────┘
```

### 1.7 Step 5: Composite Silhouette

If there are 2+ sub-paths that form a visual unit (triage Q2 = tiling), compute the **outer boundary of ALL sub-paths combined**. This is the shape you'd see if you squinted and the internal gaps disappeared.

Method: Rasterize all sub-paths to a grid. Run boundary tracing on filled pixels (ignoring internal gaps). The resulting outline = the composite silhouette.

For Auth0: the 3 sub-paths' combined silhouette = shield (flat top, curved sides, pointed bottom). Without this step, you only see "3 weird polygons."

### 1.8 Step 6: Feature Computation (ONLY the relevant ones)

By now you know: stroke vs fill, how many objects, which sub-paths belong together, basic structural classification. NOW run heavier transforms, but ONLY the ones that matter:

| SVG type | Run these | Skip these |
|---|---|---|
| Stroke-based, linear elements | Directional coverage, angular spacing | Area, width profile, wall thickness |
| Stroke-based, curved elements | Turning, centroid distance, CSS | Width profile, wall thickness, convex hull |
| Filled single shape | Curvature, width profile, convex hull, symmetry | Directional histogram, angular spacing |
| Filled with hole | Wall thickness, winding analysis | CSS, directional histogram |
| Filled composite | Composite silhouette, negative space, figure-ground | Per-sub-path curvature (do composite instead) |
| Multi-element icon (≤15) | Layer 3 relationships, grouping | Deep single-shape analysis on decorative elements |
| Complex icon (15–60+) | Full enrichment pipeline: shape class, DBSCAN, containment matrix, connected components, scored symmetry, silhouette extraction, structure heuristics | Manual per-element analysis (let engine do it) |

### 1.9 Step 7: Cross-Validation (RESONANCE)

**This is the step we skipped most often and it caused our worst failures.**

After computing features, ASK:
- Do the features AGREE with each other?
- Does the grid picture MATCH the numeric analysis?
- Are there CONTRADICTIONS between different measurements?

**Example of resonance that would have caught Auth0:**
```
Width profile says: "2 spans at some y-levels" (there's a gap)
Grid says: "filled region has internal empty space"
But the analysis said: "3 separate shield-like shapes"

CONTRADICTION: if they're 3 separate shapes, why does the grid
show ONE connected filled region with a star-shaped hole?
→ They TILE. Re-analyze as composite.
```

**Example of resonance on weather icon:**
```
Centroid distance says: "CV=0.09, it's a circle"
Turning function says: "228° of rotation, it's an open arc"
Directional histogram says: "no N or NW directions"
Rays analysis says: "5 of 8 positions occupied, gap faces SE"

ALL AGREE → high confidence. The proof came before the image.
```

#### Resonance Types

**Forward resonance** (early step informs later step): Triage says "stroke-based" → skip area metrics. This is obvious.

**Backward resonance** (later step CORRECTS earlier step): This is the critical one. You classified 3 sub-paths as "separate shapes" in triage. Then width profile shows internal gaps in the composite. GO BACK. Reclassify as composite. Re-run with dual-space analysis.

#### Resonance Triggers (when to loop back)

| Signal | What it means | Action |
|---|---|---|
| Width profile shows multi-span levels on "separate" shapes | They tile together | Reclassify as composite, run negative space |
| Grid shows connected fill but analysis says "multiple shapes" | Composite, not separate | Merge and re-analyze |
| Two transforms disagree (e.g., "convex" but hull ratio < 0.7) | One measurement is wrong or inapplicable | Check which transform applies to this SVG type |
| Numbers suspiciously close to clean values (45.4° ≈ 45°) | Export noise, not design intent | Snap to clean value |
| Numbers consistently NOT close to clean values | Designer placed by hand/eye | Keep precise values |
| Sub-path features nearly identical to another's | Repeated element | Check for transform relationship |
| Feature says "simple" but grid shows complexity | Measuring the wrong thing | Check: individual sub-paths vs composite? |

### 1.10 Step 8: Synthesis

Combine all evidence into the final analysis:

```
TEMPLATE:
  "OVERALL SHAPE: [composite silhouette description]
   POSITIVE SPACE: [what the fills form]
   NEGATIVE SPACE: [what the gaps form]
   CONSTRUCTION: [how many sub-paths, how they tile/group]
   NOTABLE FEATURES: [symmetry, curves, angles, proportions]
   PATTERNS DETECTED: [repeated elements, rotation, shared centers]
   IDENTIFICATION: [what this most likely depicts]"
```

### 1.11 Common Reasoning Traps

**TRAP 1: Figure-ground inversion.**
You analyze the filled polygons and describe what THEY compose, ignoring that the GAPS between them form the real subject. Classic: Auth0 — fills = shield, gaps = star. If you only analyze fills, you say "star" (wrong) instead of "shield with star cutout" (right).
Detection: Multiple separate spans at the same y-level in width profile → analyze gap shapes.

**TRAP 2: Sub-path = separate shape.**
SVG sub-paths within one `<path>` element are usually components of a SINGLE visual design. Don't analyze them as independent shapes.
Detection: Sub-paths share same bbox region, have matching symmetry, or combined outline forms a recognizable shape → parts of one design.

**TRAP 3: Confusing construction with visual result.**
Auth0's 3-piece construction enables fractional fill rendering. But the VISUAL RESULT is a single shield-with-star.
Detection: ALWAYS render the composite ASCII grid to see what the viewer sees.

**TRAP 4: Bézier purpose misattribution.**
Curves might be silhouette features (outer edge of overall shape) OR structural features (internal between sub-paths).
Detection: Check whether a curve is on the OUTER boundary of the composite silhouette vs INTERNAL.

**TRAP 5: Single-level analysis.**
Analyzing only individual sub-paths, or only the composite, loses information.
Detection: Always report: per-sub-path properties, composite silhouette, AND negative space.

**TRAP 6: Premature interpretation.**
Naming the shape before measuring it. "This looks like a shield" before computing composite silhouette.
Fix: Complete ALL priority 0–4 computations before any interpretation.

**TRAP 7: Wrong unit of analysis.**
Analyzing sub-paths individually when they should be analyzed as a group.
Fix: Always ask "should I analyze these INDIVIDUALLY or as a COMPOSITE?" The tiling test answers this.

---

## PART 2: TRANSFORMATION CATALOG

### Architecture: The Layer Model

```
┌──────────────────────────────────────────────────────────┐
│  LAYER 0: SVG PARSING & NORMALIZATION                    │
│  (Internal — LLM never sees raw output)                  │
│  Raw SVG → clean absolute coordinates                    │
├──────────────────────────────────────────────────────────┤
│  LAYER 1: SHAPE ANALYSIS                                 │
│  (Internal — computed per shape, per composite,          │
│   AND per negative space region)                         │
│  Clean coordinates → geometric features                  │
├──────────────────────────────────────────────────────────┤
│  LAYER 2: SPATIAL VISUALIZATION & REASONING              │
│  (LLM-facing — what the model reads and reasons about)   │
│  Features → grids, profiles, descriptions                │
├──────────────────────────────────────────────────────────┤
│  LAYER 3: MULTI-SHAPE & COMPOSITE RELATIONSHIPS          │
│  (LLM-facing — relationships between shapes AND between  │
│   positive and negative space)                           │
├──────────────────────────────────────────────────────────┤
│  LAYER 4: INVARIANCE & VALIDATION                        │
│  (Internal — pipeline QA, resonance checks)              │
│  Consistency checks, canonical orientation               │
└──────────────────────────────────────────────────────────┘
```

---

### LAYER 0: SVG Parsing & Normalization (Internal)

Always run all of these. They are mechanical.

**T0.1 — Command Expansion**
Expand shorthand SVG commands (`H`, `V`, `S`, `T`) to explicit `L`, `C`, `Q` with full coordinates. SVG has ~20 command types; analysis needs 4 (`M`, `L`, `C`, `Q`). `H`/`V` hide one coordinate. `S`/`T` hide a control point (reflected from previous segment). Without expansion, curve geometry is partially invisible.

**T0.2 — Relative → Absolute Coordinate Resolution**
Convert lowercase (relative) commands to absolute. Each relative offset depends on ALL previous commands — a single tracking error propagates to every subsequent coordinate.

**T0.3 — Arc → Bézier Approximation**
Convert elliptical arc commands (`A`) to cubic bézier segments. Arc parameters (rx, ry, x-rotation, large-arc-flag, sweep-flag) are the hardest SVG construct. Library: `svgpathtools`.

**T0.4 — Bézier Sampling (Boundary Point Generation)**
Sample N evenly-spaced points along each path boundary using **arc-length parameterization** (not uniform `t`). N = 500–2000 per path (8–12 per segment). **Adaptive sampling:** increase N where curvature is high (use second derivative magnitude to detect sharp bends), decrease where curvature is low (near-straight). This gives the LLM a pointcloud silhouette: instead of reasoning about cubic polynomial math, it reads coordinates tracing the actual curve. `svgpathtools` provides `segment.point(t)` for parametric evaluation.

**T0.5 — Sub-Path Extraction**
Split compound paths (`M...Z m...Z m...Z`) into separate closed sub-paths. **Count sub-paths immediately — it's the first structural signal.**

**T0.6 — Winding Direction Detection**
Signed area via shoelace formula. Positive = CCW, Negative = CW. Combined with fill-rule, determines which regions are "inside" vs "outside."

**T0.7 — Transform Resolution**
Apply CSS/SVG transforms (`translate`, `rotate`, `scale`, nested `<g>` group transforms). Compose transform matrices top-down through DOM tree.

**T0.8 — ViewBox Normalization — DEFERRED**
Normalize to `[0,1]×[0,1]` space. Only matters for cross-SVG comparison. Re-add when needed.

---

### LAYER 1: Shape Analysis (Internal — computed, then summarized for LLM)

**Apply to:** each individual sub-path, the composite silhouette, AND negative space regions.

#### Boundary Descriptors

**T1.1 — Curvature Profile** ★★★
Signed curvature κ(t) at each boundary point. THE shape fingerprint — identical curvature profiles = identical shapes. Distinguished Bélo's smooth peak from standard heart's V-notch. Summarize as: peak locations, magnitudes, smooth vs sharp, symmetry.

**T1.2 — Inflection Point Detection** ★★
Points where curvature crosses zero (convex↔concave). Count and positions define topological character. Circle: 0. Figure-8: 4. Heart: 2–4.

**T1.3 — Turning Function** ★★ PROVEN
Cumulative tangent angle θ(s) as function of arc-length. Cheap instant classifier: total ≈ 360° = closed loop, ≈ 180° = semicircle, ≈ 0° = straight. Proven: sun arc = 227.6° (⅔ circle open arc), cloud = 358.9° (closed loop). Fastest open-vs-closed test. Essential for stroke-based SVGs.

**T1.4 — Centroid Distance Signature** ★★ PROVEN
Distance from centroid to each boundary point. CV (std/mean) is cheapest circle detector: <0.1 = circular, 0.1–0.3 = elliptical, >0.3 = complex. Proven: sun arc CV=0.090 confirmed circular without fitting math. Works on OPEN arcs where area-based circularity is meaningless.

**T1.5 — Fourier Shape Descriptors** ★ INTERNAL ONLY
FFT of boundary coordinates. Tested: numbers like "H2=0.12" meaningless to LLM without lookup table. Keep as internal fingerprint for shape comparison/matching only.

**T1.6 — Directional Coverage Analysis** ★★ PROVEN
Quantize boundary directions into 8–16 angular bins. Proven on weather icon in TWO ways: (1) Sun arc: 0% in N↑/NW↖ = only covers exposed half. (2) Rays: 176° angular gap = rays on one side only = partial occlusion signal. THE discriminator for "partly cloudy" vs "full sun."

**T1.7 — Beam Angle Statistics (BAS)** — CUT
Tested. Never produced an insight simpler transforms didn't already give. Convex hull provides same info more clearly.

**T1.8 — Curvature Scale Space (CSS)** ★★ PROVEN
Progressively smooth boundary, count inflection points at each scale. Points persisting at high σ = fundamental features. Proven: cloud = 4 stable inflections = "2-bump cloud with flat base." The inflection count at max σ is a powerful complexity measure.

#### Region Descriptors

**T1.9 — Width Profile Function** ★★★ MOST LLM-FRIENDLY
Horizontal cross-section width at each y-level. **CRITICAL ENHANCEMENT:** Report BOTH filled spans AND gap spans, and count spans per level.
- Always 1 span → solid shape
- 2+ spans → internal gaps/cutouts → MANDATORY negative space analysis

**T1.10 — Wall Thickness Profile** ★★★
For shapes with holes: distance between outer and inner boundaries at each angle. KEY Bélo discriminator (top: 1.7, sides: 11.4, bottom: 6.7 = person/letter-A pattern).

**T1.11 — Convex Hull & Convexity** ★★
Convexity ratio = shape_area / hull_area. Circle: 1.0, star: ~0.5. Gap between hull and shape reveals concavity locations.

**T1.12 — Medial Axis / Skeleton** ★★ CONDITIONAL: FILLED SHAPES ONLY
Tested on weather icon (stroke-based): garbage (64 junctions from artifacts). Strokes already ARE the skeleton. For FILLED shapes: gives clean stick-figure topology.

**T1.13 — Signed Distance Field (SDF)** — INTERNAL ONLY
Useful internally for finding shape "centers" and measuring depth. Not useful as LLM-facing description (duplicates ASCII grid).

#### Global Scalar Descriptors

**T1.14 — Basic Geometric Properties** ★★★
Area, perimeter, bounding box, centroid, aspect ratio, sub-path count, segment type counts, orientation (PCA major axis angle), curvature variance (smooth arc vs jagged). Cheap, always compute.

**T1.15 — Circularity** ★ — `C = 4π·area/perimeter²`. Circle=1.0.

**T1.16 — Rectangularity** — `R = area / min_bounding_rectangle_area`. Rectangle=1.0.

**T1.17 — Eccentricity** — Major/minor axis ratio. Circle=1.0. Also gives principal axis orientation.

**T1.18 — Hu Moment Invariants** ★★ — 7 numbers invariant to translation, rotation, scale. Use log-transform (`sign(h) × log10(1 + |h|)`) before distance computation. Primary use: shape similarity matching via DBSCAN clustering. Two elements with Hu distance < 0.5 look alike regardless of position/size. Essential for detecting repeated decorative elements (spots, scales, feathers) in complex SVGs.

**T1.19 — Zernike Moments** ★ — Orthogonal complex moment basis. More discriminating than Hu. Internal.

**T1.20 — Symmetry Detection** ★★★
Test bilateral symmetry by reflecting all element centroids across candidate axes and measuring match quality.
1. Test vertical axis (x = viewBox_width/2) first — most common
2. Test horizontal, then diagonals (45°, 135°)
3. Symmetry score = mean(1 - distance/eps) for matched pairs
4. Output: best axis, score (0–1), list of mirror pairs, on-axis elements
Also test rotational symmetry: check if centroid angle set is invariant under 360°/n for n = 2,3,4,5,6,8.

**T1.21 — Corner / Vertex Detection (VTracer method)** ★★
Adopted from VisionCortex's VTracer: compute the signed rotation angle θ ∈ (−π, π] between consecutive edge vectors eᵢ and eᵢ₊₁. Track accumulated "angle displacement" from the last detected corner. When |displacement| exceeds a threshold → splice point (corner).

```
For each vertex i:
  θᵢ = signed_angle(eᵢ, eᵢ₊₁)    # rotation needed to align eᵢ to eᵢ₊₁
  displacement += θᵢ
  IF |displacement| > corner_threshold:
    mark as corner, reset displacement = 0
```

O(n), configurable sensitivity (lower threshold = more corners = tighter fit). Proven at gigapixel scale in VTracer. For polygons: corners ARE the shape — report angles at each corner. For curves: splice points define where Bézier segments should break.

**T1.22 — Triangular Area Representation (TAR)** — CUT
Sign depends on winding direction → inverted results for CW paths. Redundant with convex hull.

**T1.23 — Shape Class Auto-Labeling** ★★★ NEW
Thresholded classification from T1.14–T1.17, computed per element:
```
IF circularity > 0.85 AND aspect_ratio 0.8–1.2  → "circular"
IF circularity > 0.75 AND aspect_ratio outside 0.8–1.2  → "elliptical"
IF rectangularity > 0.85  → "rectangular"
IF convexity > 0.9 AND num_vertices == 3  → "triangular"
IF aspect_ratio > 5 OR aspect_ratio < 0.2  → "linear"
ELSE  → "organic"
```
This gives each element a one-word label for quick scanning. "15 elements: 2 circular, 1 elliptical, 12 organic" is instantly useful.

**T1.24 — Elliptic Fourier Descriptors (EFD)** ★ INTERNAL
Decompose closed contour into frequency components. Low-frequency = gross shape, high = fine detail. Truncating to first 8–12 harmonics gives a smooth shape approximation — useful for matching organic shapes (leaves, animal silhouettes) across scale and rotation. More discriminating than Hu for organic shapes.

---

### LAYER 2: Spatial Visualization & Reasoning (LLM-Facing)

These are the representations you directly read and reason about.

**T2.1 — ASCII Grid** ★★★ PRIMARY — ALWAYS FIRST
Rasterize shapes onto text grid. Compute TWO grids:
1. POSITIVE grid: `█` = filled, `·` = empty
2. NEGATIVE grid (inverted, within bbox): `█` = empty-inside-bbox, `·` = filled
Resolution: 32×32 for standard 24×24 icons.

**T2.2 — Region Map** ★★
Divide canvas into 4×4 grid. Per region: fill percentage + density label. Synthesis rules: center sparse while surround dense → "central cutout pattern."

**T2.3 — Multi-Resolution Description** ★★
Describe at multiple zoom levels: COARSE (2×2) → MEDIUM (4×4) → FINE (8×8) → DETAIL (16×16).

**T2.4 — Macro Trajectory Narrative** ★★
Natural language description of composite silhouette boundary direction.

**T2.5 — Composite Silhouette Description** ★★★ CRITICAL
Trace outer boundary of all filled pixels combined (ignoring internal gaps). Method: rasterize → morphological close → boundary trace → describe.

**T2.6 — Negative Space Description** ★★★ CRITICAL
Analyze empty regions INSIDE composite bounding box. Count connected components, describe each. Trigger rule: if width profile shows 2+ spans at ANY y-level.

**T2.7 — Figure-Ground Report** ★★★ CRITICAL
```
OUTPUT:
  "POSITIVE SPACE (filled): [description]
   NEGATIVE SPACE (empty within bbox): [description]
   COMPOSITE SILHOUETTE: [combined outer boundary]
   FIGURE-GROUND: The [positive/negative/both] carries the primary design.
   [If both: describe relationship — e.g., 'star cut from shield']"
```
Decision: higher geometric regularity = usually the designed element. Default: report both.

---

### LAYER 3: Multi-Shape & Composite Relationships (LLM-Facing)

#### Standard Inter-Shape Relationships

| ID | Relationship | Method | Description |
|---|---|---|---|
| T3.1 | Containment | Winding number on discretised contour; test 5 points (centroid + 4 bbox midpoints), majority vote | Shape A inside Shape B? Output as containment matrix. |
| T3.2 | Distance | Min distance between pointclouds | Closest points between shapes. Output as proximity matrix. |
| T3.3 | Alignment | Compare centroids | Shapes share center x/y? |
| T3.4 | Relative Size | Area ratio from T1.14 | Shape A is 3× larger than B |
| T3.5 | Relative Position | Centroid difference + quadrant | B is above A, offset 2 units right |
| T3.6 | Overlap | Bbox intersection + pointcloud test | Shapes intersect? Overlap area? |
| T3.7 | Grouping (DBSCAN) | DBSCAN on centroids, eps = 8–12% of viewBox diagonal, min_samples=2 | Which shapes form spatial clusters? Noise label = isolated element. |
| T3.8 | Symmetry | Mirror pair matching from T1.20 | Shapes A and B are mirror images about axis X |
| T3.9 | Repetition | Hu-moment clustering from T1.18 | Shape B appears 5 times in regular pattern |
| T3.10 | Topology | Containment matrix + proximity | "2 circles inside larger circle" = face pattern |

**T3.7 output format:**
```
Cluster A [5 elements]: circular ring at (128, 95), all circular shapes
Cluster B [3 elements]: horizontal row at y≈160, mixed organic+linear
Isolated: E1 (body outline), E7 (head outline)
```

For each cluster, compute: cluster centroid, radius (max distance from center to any member), member count, dominant shape class (from T1.23), internal arrangement (linear/circular/scattered).

#### Composite Sub-Path Relationships

**T3.11 — Sub-Path Tiling Classification** ★★★
Do sub-paths TILE (adjacent, filling region together), NEST (one inside another), or SEPARATE?
Method: Check bbox overlaps, shared edges, vertex coincidence.

**T3.12 — Gap Analysis Between Sub-Paths** ★★★
For tiling sub-paths: count gap regions, measure areas, check if gaps form recognizable pattern.
Auth0: 5 triangular gaps forming a star = THE identification feature.

**T3.13 — Construction Purpose Inference** ★★
Why split into these sub-paths? 2 halves → half-fill rendering. 3 pieces → fractional fill. Hole + outer → color independence.

#### Pattern Detection (for stroke-based and multi-element SVGs)

**T3.14 — Repeated Element Detection** ★★★
Cluster sub-paths by (length, type, segment count). If cluster size ≥ 2: check for common transform pattern.
- All share a center point? → radial/rotation pattern
- All offset by same delta? → grid/array pattern
- All reflected about an axis? → mirror pattern
Output: template element + list of transforms.

**T3.15 — Shared Center / Origin Detection** ★★★
For each cluster of related elements: compute centroids/midpoints. Do they all sit at equal distance from a common point? Does a fitted circle's center match the cluster center?
If yes → group under one `<g transform="translate(cx,cy)">`.

**T3.16 — Angular Spacing Analysis** ★★
For radially-arranged elements: compute angle between each pair. If spacing is within ±2° of a clean multiple (15°, 30°, 45°, 90°) → rotation tool was used → snap to clean angle.

**T3.17 — Occlusion / Incompleteness Detection** ★★
When an arc covers < 360° AND rays/spokes cover < 360° AND the missing portions face the same direction → element is partially hidden behind something.
Weather icon: ⅔ circle arc + rays on one side only → sun partially hidden by cloud.

#### Topology & Structure Detection (for complex SVGs with 20+ elements)

**T3.18 — Connected Component Graph** ★★★ NEW
Build a graph: elements = nodes, edges = containment (from T3.1) OR proximity < 2% of viewBox diagonal OR shared boundary segment. Run connected components. Each component is a visual unit.
```
Component 1 [E1, E2, E3, E4, E5, E6]: head group (E1 contains E2–E6)
Component 2 [E7, E8, E9]: body group (E7 contains E8, E9)
Component 3 [E12]: isolated (tail?)
```
This tells the LLM "these elements belong together" before any semantic reasoning.

**T3.19 — Structural Pattern Report** ★★ NEW
Report recurring geometric arrangements as factual descriptions — NOT semantic guesses. The LLM interprets; the engine describes.

**Report format (geometry only, no naming):**
```
PATTERN: 1 large containing element (circular, area=X)
  + 2 symmetric elements in upper quadrant (circular, small)
  + 1 element in lower-center (organic, small)
  Bilateral symmetry score: 0.91

PATTERN: radial arrangement of 8 similar elements
  around central element, angular spacing ~45°
  Rotational symmetry: 8-fold

PATTERN: 1 large element (organic, high aspect ratio)
  + appendage-like protrusions from boundary
  + bilateral symmetry score: 0.82
```

**Do NOT write:** "possible face structure" or "possible flower." That's premature interpretation (Trap 6). The LLM has the guide and will recognise the pattern itself. Injecting semantic labels anchors it on a guess the engine isn't qualified to make.

**T3.20 — Composite Silhouette Extraction** ★★★ NEW
Compute outer boundary of ALL elements combined as a single closed path (the silhouette — what you'd see if the SVG were a solid black shadow).
1. Rasterize all elements to binary grid (2× viewBox resolution)
2. Morphological close (bridge gaps < 2% of viewBox diagonal)
3. Find outer contour (marching squares or boundary trace)
4. Simplify contour (Ramer-Douglas-Peucker, epsilon = 1% of diagonal)
5. Classify silhouette using T1.23 (shape class) and T1.20 (symmetry)

Output: silhouette bbox, aspect ratio, circularity, convexity, concavity locations (protrusions = ears? limbs? horns?), shape class.

This is the computational version of T2.5 but done by the geometry engine instead of by the LLM — essential when there are too many elements for LLM rasterisation.

**T3.21 — Visual Stacking Tree** ★★★ NEW
Inspired by VisionCortex Impression's "image tree" and painter's model — background is laid down first, then objects, then details. For SVGs, build a parent-child hierarchy from z-order (SVG document order) + containment (T3.1) + relative area (T1.14).

Construction algorithm:
1. Sort elements by z-order (SVG document order = painting order)
2. For each element, find its containing parent via containment matrix (T3.1)
3. If no parent → root-level element (background layer)
4. If contained by multiple elements → parent is the smallest container (nearest ancestor)
5. Within each parent, order children by z-order

Output format:
```
VISUAL STACKING TREE:
  E1 [organic, LARGE, z:0] — background/body
  ├── E2 [circular, MEDIUM, z:1] — inside E1, upper-left
  ├── E3 [circular, MEDIUM, z:2] — inside E1, upper-right (mirror of E2)
  ├── E4 [organic, SMALL, z:3] — inside E1, center
  │   └── E5 [circular, SMALL, z:4] — inside E4
  └── E6 [linear, SMALL, z:5] — inside E1, lower
  E7 [organic, MEDIUM, z:6] — root level, overlaps E1
```

Why this matters: a flat list of 30 elements is hard to reason about. A tree with 4 root nodes, each containing 5-8 children, is a scene graph. The LLM reads it as "body contains eyes, nose, mouth" instead of "E1, E2, E3, E4, E5, E6 are all separate things."

This mirrors how Impression walks its image tree top-down "like a painter laying down background first, then overlaying objects, then adding details." Same principle: hierarchical structure is more meaningful than flat enumeration.

---

### LAYER 4: Invariance & Validation (Internal)

**T4.1 — Canonical Orientation Normalization**
Rotate to standard orientation before computing orientation-dependent features.

**T4.2 — Rotation-Invariant Feature Set**
Bundle: area, perimeter, circularity, convexity, Hu moments, inflection count, corner count, symmetry order.

**T4.3 — Multi-Orientation Consistency Check**
Run at 0°, 90°, 180°, 270°. Verify descriptions are equivalent.

**T4.4 — Hilbert Spatial Index**
Index grid cells for O(log n) spatial queries. Performance optimization.

**T4.5 — Dual-Space Consistency Check** ★★
Verify positive + negative + silhouette tell a CONSISTENT story. If they contradict → flag and loop back.

---

## PART 3: SEMANTIC SVG RECONSTRUCTION

The analysis pipeline doesn't just identify shapes — it produces enough information to reconstruct the SVG in semantically meaningful form. This is the output stage.

### 3.1 The Three-Stage Architecture

```
RAW SVG (export garbage: 1 path, baked coordinates, no structure)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1: GEOMETRIC ANALYSIS  (no ML needed)        │
│  Pure math. Deterministic. Fast.                    │
│                                                     │
│  • Parse SVG, extract sub-paths                     │
│  • Circle detection from cubic sequences            │
│  • Repeated element detection                       │
│  • Shared center / transform detection              │
│  • Line-from-cubic detection (collinear controls)   │
│  • Shared style detection                           │
│  • Sub-path relationship classification             │
│                                                     │
│  OUTPUT: structured geometric facts (not opinions)  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: LLM REASONING  (this document guides you) │
│  WHY was it built this way? What's the intent?      │
│                                                     │
│  • "5 cubics with CV=0.09 from a center" →          │
│    "this is a circle arc"                           │
│  • "5 identical lines at 45° intervals" →           │
│    "this is 1 ray, rotated"                         │
│  • "arc + rays share center" →                      │
│    "one object: a sun"                              │
│  • "3 ray angles missing" →                         │
│    "intentional: sun is occluded"                   │
│  • "45.4° measured" →                               │
│    "designer meant 45°, export added noise"         │
│  • "6 organic cubics, no symmetry" →                │
│    "hand-drawn, can't simplify, keep as path"       │
│                                                     │
│  OUTPUT: semantic decisions                         │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3: CODE GENERATION                           │
│  Turn understanding into clean SVG + CSS.           │
│                                                     │
│  • Detected circle → <circle> + dasharray           │
│  • Repeated element → <defs> + <use> + rotate()     │
│  • Shared center → <g transform="translate(cx,cy)"> │
│  • Shared styles → CSS classes + variables           │
│  • Noise → round to clean values                    │
│  • Irreducible curves → keep as <path>, label it    │
│  • Add IDs, aria labels, comments                   │
│                                                     │
│  OUTPUT: semantic SVG + component manifest           │
└─────────────────────────────────────────────────────┘
```

### 3.2 Enrichment Summary (what the geometry engine hands to the LLM)

When the geometry engine (Stage 1) finishes, it produces this structured summary. This is the actual text block injected alongside the raw SVG into the LLM context. ~1,200 tokens. The LLM reads it in one pass instead of measuring things itself.

```
=== VECTORSIGHT ENRICHMENT (auto-computed) ===

ELEMENTS: [count] paths, [count] sub-paths total

PER-ELEMENT (sorted by area, top N):
  E1: [SHAPE_CLASS], area=[n], bbox([x1],[y1],[x2],[y2]),
      centroid([x],[y]), circ=[n], convexity=[n], aspect=[n]
  E2: ...

CONTAINMENT:
  E1 contains: E2, E3, E4, ...
  E7 contains: E8, E9, ...

SPATIAL CLUSTERS (DBSCAN, eps=[n]% of diagonal):
  Cluster A [E2,E3,E4]: [description] inside E1
  Cluster B [E8,E9]: [description]
  Isolated: E1 (body), E12 (tail)

SYMMETRY: [type] about [axis] (score [0-1])
  Pairs: (E2↔E3), (E4↔E5), ...
  On-axis: E1, E7, ...
  [Rotational: n-fold if detected]

SIZE TIERS:
  LARGE: E1, E7 (structural outlines)
  MEDIUM: E2–E6 (features)
  SMALL: E10–E15 (details)

SHAPE SIMILARITY (Hu moment matching):
  Identical: (E2, E3) — mirrored pair
  Similar: (E8, E9, E10, E11) — repeated decoration
  Unique: E1, E7, E12


CONNECTED COMPONENTS:
  Component 1 [E1,E2,E3,E4,E5,E6]: [description]
  Component 2 [E7,E8,E9]: [description]
  ...

VISUAL STACKING TREE:
  E1 [shape_class, SIZE_TIER, z:0] — root
  ├── E2 [shape_class, SIZE_TIER, z:1] — inside E1, [position]
  ├── E3 [shape_class, SIZE_TIER, z:2] — inside E1, [position]
  │   └── E5 [shape_class, SIZE_TIER, z:4] — inside E3
  └── E6 [shape_class, SIZE_TIER, z:5] — inside E1, [position]
  E7 [shape_class, SIZE_TIER, z:6] — root, overlaps E1

SILHOUETTE: [shape class], bbox([...]), aspect=[n], circ=[n],
  convexity=[n], concavities at ([x],[y]) and ([x],[y])

STRUCTURAL PATTERNS:
  [geometric arrangement descriptions from T3.19, if any detected]

COLOUR: [list colours used, note if monochrome/brand-coloured]
Z-ORDER: [SVG document order = intended layering]

=== END ENRICHMENT ===
```

**Token budget:** ~1,200 tokens for 15 elements. Can compress to ~800 by reporting only top-N elements by area and skipping uniques in similarity. For 60+ elements, report top 10 by area and summarize the rest by tier.

**Colour note:** Shape classification from geometry alone is strong for distinctive shapes. Brand identification often requires colour as a disambiguating signal — include it as metadata.

**Z-order note:** SVG document order is the closest proxy for intended layering. Later paths render on top. Preserve this as metadata — without it, foreground/background is ambiguous.

### 3.3 What's Pure Geometry vs What Needs LLM Reasoning

**Computable (no LLM needed):**
- Circle detection: centroid distance CV < 0.1 on sampled points = circular
- Ellipse detection: CV low but distance varies sinusoidally
- Line detection: cubic bézier control points collinear with endpoints
- Repeated elements: cluster by (length, type), check angular spacing
- Shared center: all midpoints equidistant from a common point
- Shared styles: extract, find common properties → CSS class
- Clean-number snapping: value within ±1° of 45° multiple → snap

**Needs LLM reasoning:**
- Object grouping: "rays + arc share center" → "that's one object called 'sun'" (not clock, compass, asterisk)
- Occlusion reasoning: "3 missing rays + partial arc on same side" → "something hides them"
- Noise vs intent: "45.4° in an icon context → designer meant 45°"
- Foreground/background: "cloud has later z-order and overlaps sun → cloud is foreground"
- Semantic naming: "closed organic 2-bump curve" → "cloud"
- Simplification limits: "this organic curve can't be reduced to primitives, keep it"

**The geometry reduces 500 characters of path data to ~15 structured facts. The LLM reasons over 15 facts instead of 500 characters. This is what makes it tractable.**

### 3.4 Representation Decisions

After analysis, you must make representation choices. These are design philosophy decisions — not computable. Consider the options explicitly.

#### Circular arcs: three valid representations

| Option | SVG code | When to use |
|---|---|---|
| Dashed circle | `<circle r="2" stroke-dasharray="..."/>` | Designer intended a full circle with hidden portion. More editable. |
| SVG arc command | `<path d="M... A r r 0 1 1 ..."/>` | Designer drew an arc. More faithful to original construction. |
| Keep original cubics | `<path d="M... C... C... C..."/>` | Arc is hand-drawn, wobbly, or exact reproduction required. |

Decision criterion — centroid distance CV:
- CV < 0.05: safe to use `<circle>` or `A` arc (proven perfect circle)
- CV 0.05–0.15: use `A` arc (approximately circular, honest about it)
- CV > 0.15: keep cubics (not really circular)

#### Repeated elements: template vs explicit

| Signal | Representation |
|---|---|
| All instances identical AND clean angular/linear spacing | `<defs>` + `<use>` + `transform` |
| Similar but not identical (different lengths, slight variations) | Explicit elements, note the pattern in a comment |
| Only 2 instances | Probably not worth `<defs>/<use>` overhead |

#### Noise cleanup: when to round values

**Round when:**
- Value within ±1° of a multiple of 15°, 30°, 45°, or 90°
- Value within ±0.1 of an integer or common fraction (0.5, 0.25)
- ALL instances in a repeated cluster converge near the same clean value
- SVG is an icon (icons use grids and clean values)

**DON'T round when:**
- Value consistently NOT near a clean number
- Only 1 instance (can't distinguish intent from noise)
- SVG is illustration or hand-drawn artwork
- Rounding would change the visual (2° matters at large radius)

#### Grouping: what gets a `<g>`

- Elements sharing a center point → `<g transform="translate(cx,cy)">`
- Elements forming one conceptual object → `<g id="object-name">`
- Elements with identical styling → `<g class="shared-style">`
- Foreground vs background → separate `<g>` blocks, foreground AFTER background

#### What stays as `<path>`

Some things can't be simplified: organic hand-drawn curves, complex irregular polygons, shapes with intentional irregularity. Say so explicitly. A comment like `<!-- organic shape, not reducible to primitives -->` is honest and useful.

#### Style extraction to CSS

- All elements sharing same stroke/fill → CSS class with variables
- `stroke-linecap: round` on short lines → this is WHY rays look like pills (document this)
- `stroke-dasharray` on circles → this is HOW partial arcs are shown (document the math)
- CSS variables (`--color`, `--weight`) → change once, everything updates

### 3.5 Semantic SVG Output Template

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- [Icon name]: Reconstructed from geometric analysis -->
<svg viewBox="0 0 W H" xmlns="http://www.w3.org/2000/svg"
  role="img" aria-labelledby="title desc">
  <title id="title">[Name]</title>
  <desc id="desc">[What it depicts]</desc>

  <style>
    :root { --color: #000; --weight: 2; }
    .stroke { fill: none; stroke: var(--color);
              stroke-width: var(--weight);
              stroke-linecap: round; stroke-linejoin: round; }
    /* [Document any dasharray/dashoffset math] */
  </style>

  <defs>
    <!-- Reusable elements (template for repeated parts) -->
  </defs>

  <!-- [GROUP 1: name, role, construction logic] -->
  <g id="[name]" class="[style]" transform="translate(cx cy)">
    <!-- [Element: what it is, how it was derived] -->
  </g>

  <!-- [GROUP 2: name, relationship to group 1] -->
  <g id="[name]" class="[style]">
    <!-- [If irreducible: say so] -->
    <!-- organic shape, not reducible to primitives -->
    <path id="[name]" d="..." />
  </g>
</svg>
```

### 3.6 Component Manifest (CSV)

Alongside the SVG, output a CSV describing the construction parameters:

```
group,id,element,construction,transform,notes
[group],[id],[svg element],[how it was built],[what transform],[design intent]
```

Plus a DESIGN PARAMETERS section with derived values:
```
param,value,unit,derivation
sun_radius,2.0,viewBox units,Circle fit CV=0.090
ray_spacing,45,degrees,Measured 44.1°–45.4° → snapped (icon grid)
arc_visible,228,degrees,Turning function = 227.6°
```

---

## PART 4: SVG CREATION & EDITING

### 4.1 The Core Problem: LLMs Can't Do Coordinate Math

LLMs are bad at arithmetic. They can reason "the eyebrow goes above the eye with a small gap" but they cannot reliably compute that `cy_eyebrow = 6.5 - 0.5 - 0.2 = 5.8`. Compound calculations, Bézier control points, proportional scaling — the LLM is guessing confidently, not computing.

The enrichment solves the **reading** problem (understanding existing SVGs). But asking the LLM to **write** precise coordinates is asking it to do what it's worst at.

**Solution: the LLM never writes coordinates.** It writes **spatial intent** — relative descriptions of what goes where. The geometry engine resolves intent into coordinates, the same way it resolves coordinates into descriptions for analysis.

```
ANALYSIS (already built):
  Raw SVG → engine → enrichment → LLM reads spatial descriptions

CREATION (mirror image):
  LLM writes spatial intent → engine → SVG coordinates → validation
```

Same vocabulary. Opposite direction. The engine handles all the numbers.

### 4.2 Spatial Intent Language

The LLM outputs structured spatial descriptions. The engine resolves them to coordinates. The vocabulary mirrors the enrichment summary — every term the LLM reads in analysis, it can write in creation.

```
=== SPATIAL INTENT ===

CANVAS: viewBox 24 24

ELEMENT body
  SHAPE: organic
  FILL: 65% of canvas
  POSITION: center of canvas
  PATH: [raw path data OR reference to extracted component]

ELEMENT head
  SHAPE: circular
  SIZE: 25% of body area
  POSITION: upper-right of body, attached to body boundary
  PADDING: 0% (touching body edge, may protrude)

ELEMENT eye
  SHAPE: circle
  SIZE: 4% of head area
  POSITION: inside head, upper-forward quadrant (75% x, 35% y)
  PADDING: 10% from head boundary minimum
  DEPTH: above head

ELEMENT eyebrow
  SHAPE: arc
  SIZE: width 130% of eye width, height 30% of eye height
  POSITION: above eye, gap 20% of eye height, centered on eye x
  DEPTH: above eye

MIRROR: eye, eyebrow
  AXIS: vertical center of head

ELEMENT beak
  SHAPE: triangular
  SIZE: 8% of head area
  POSITION: center-right of head, on head boundary, protruding 50% outside

STYLE:
  fill: #000000
  stroke: none

=== END INTENT ===
```

### 4.3 What the Engine Does With Intent

Each intent line maps to a geometric operation:

| Intent | Engine operation |
|---|---|
| `FILL: 65% of canvas` | Compute bbox: `sqrt(0.65) × 24 ≈ 19.4` per side, center in viewBox |
| `SIZE: 25% of body area` | `sqrt(0.25 × body_area / π) = radius` for circles |
| `POSITION: inside head, upper-forward (75% x, 35% y)` | `x = head_x1 + 0.75 × head_w`, `y = head_y1 + 0.35 × head_h` |
| `PADDING: 10% from boundary` | Verify min distance from element edge to container edge ≥ 10% of container size |
| `MIRROR: eye AXIS: vertical center` | Reflect all mirrored elements: `x_new = 2 × axis_x - x_old` |
| `above X, gap 20% of X height` | `y = X_top - gap - element_height` |
| `attached to boundary, protruding 50%` | Place centroid on container edge, half inside, half outside |
| `DEPTH: above head` | Place later in SVG document order than head |

The engine outputs final SVG with exact coordinates. The LLM never computed a single number.

### 4.4 Three Modes

**ANALYSIS:** existing SVG → engine → enrichment summary → LLM reads
**EDITING:** existing SVG + enrichment + instruction → LLM writes intent for new/changed elements → engine resolves against existing elements → validated SVG
**CREATION:** text description → LLM writes full intent → engine resolves all coordinates → validated SVG

#### Editing (existing SVG + enrichment available)

The LLM reads the enrichment, identifies the target region, and writes intent only for the new/changed elements. Existing elements provide the coordinate frame.

```
User: "Add an eye to this bird"

LLM reads enrichment:
  E1 (body): organic, area=280, bbox(3,4)-(21,20)
  Structural pattern: protrusion upper-right of body
  Protrusion bbox: (17,3)-(23,9) ← this is the head

LLM writes intent:
  ELEMENT eye
    SHAPE: circle
    SIZE: 4% of protrusion area
    POSITION: inside protrusion, upper-forward (70% x, 35% y)
    FILL: white
    DEPTH: above E1

Engine resolves:
  protrusion area = 6 × 6 = 36
  radius = sqrt(0.04 × 36 / π) = 0.68
  x = 17 + 0.70 × 6 = 21.2
  y = 3 + 0.35 × 6 = 5.1
  → <circle cx="21.2" cy="5.1" r="0.68" fill="white"/>

Engine validates:
  Inside protrusion? ✓ (winding number)
  Padding ≥ 10%? ✓ (min edge distance = 1.12 > 0.6)
  Size tier SMALL? ✓
```

#### Creation (from scratch)

The LLM plans the entire composition as a hierarchy of spatial intent. The engine resolves all coordinates at once, maintaining consistency.

```
User: "Create a simple bird icon"

LLM writes intent:
  CANVAS: viewBox 24 24

  ELEMENT body
    SHAPE: organic (horizontal oval)
    FILL: 60% of canvas
    POSITION: center, slight left offset (45% x, 50% y)

  ELEMENT head
    SHAPE: circular
    SIZE: 22% of body area
    POSITION: upper-right of body, protruding 30% beyond body boundary

  ELEMENT eye
    SHAPE: circle
    SIZE: 4% of head area
    POSITION: inside head, upper-forward (65% x, 35% y)
    PADDING: 10% from head boundary

  ELEMENT beak
    SHAPE: triangular
    SIZE: width 40% of head width, height 25% of head height
    POSITION: right of head, on head boundary, pointing right
    PROTRUDE: 70% outside head

  ELEMENT wing
    SHAPE: organic (curved triangle)
    SIZE: 30% of body area
    POSITION: center of body, slight back (40% x, 50% y)
    EXTENDS: below body boundary by 20%

  ELEMENT tail
    SHAPE: triangular
    SIZE: 12% of body area
    POSITION: left of body, on body boundary, pointing left-up
    PROTRUDE: 80% outside body

  DEPTH ORDER: body, wing, head, eye, beak, tail
  STYLE: fill #1DA1F2, eye fill #FFFFFF, beak fill #FFAD1F

Engine resolves entire composition:
  1. body bbox from 60% fill + position → (4,5) to (19,19)
  2. head from 22% of body + upper-right → circle at (20, 7) r=2.8
  3. eye from 4% of head + upper-forward → circle at (21.2, 5.9) r=0.36
  4. beak from 40%×25% of head + right boundary → triangle at (22.8, 7)
  5. wing from 30% of body + center-back → organic shape
  6. tail from 12% + left boundary → triangle at (2.5, 8)
  → Complete SVG with all coordinates
```

### 4.5 Component Reuse (for fonts and repeated patterns)

When editing or creating within an established style (same font, same icon set), the LLM can reference **extracted components** from analysed SVGs instead of describing shapes from scratch.

The engine analyses reference SVGs and builds a component library:

```
=== COMPONENT LIBRARY (extracted from font analysis of F, O, A, S) ===

COMPONENT v_stem
  SOURCE: letter F, segment 1
  PATH: M0,0 L3,0 L3,20 L0,20 Z
  PROPERTIES: width=3, height=variable, rectangular

COMPONENT h_bar
  SOURCE: letter F, segment 2
  PATH: M0,0 L12,0 L12,2.1 L0,2.1 Z
  PROPERTIES: width=variable, height=2.1 (contrast ratio 1.4:1 vs stem)

COMPONENT serif
  SOURCE: letter F, terminals
  PATH: C(0,0, 1.2,0, 1.5,0.8, 1.5,1.2)
  PROPERTIES: bracketed, 1.5 wide, 1.2 tall

COMPONENT bowl_R
  SOURCE: letter O, right half
  PATH: [extracted cubic Bézier sequence]
  PROPERTIES: curvature profile [...], smooth, overshoot 0.2

PARAMETERS:
  stem_width: 3
  contrast: 1.4
  cap_height: 20
  x_height: 14
  serif_size: 1.5 × 1.2
  overshoot: 0.2

=== END LIBRARY ===
```

The LLM then composes new letters by selecting and placing components:

```
LLM intent for letter "B":

  CANVAS: viewBox matching font metrics (cap_height=20)

  PLACE v_stem
    AT: left edge
    HEIGHT: cap_height

  PLACE bowl_R
    ATTACH: right side of v_stem, top half
    SCALE: height to 48% of cap_height
    FLIP: none

  PLACE bowl_R
    ATTACH: right side of v_stem, bottom half
    SCALE: height to 52% of cap_height
    FLIP: none

  PLACE serif
    AT: v_stem, top-left terminal

  PLACE serif
    AT: v_stem, bottom-left terminal

  PLACE h_bar
    AT: junction between upper and lower bowls
    WIDTH: stem_width (flush with stem)

  JOIN: smooth tangent-continuous curves at all component junctions
```

The engine:
1. Looks up each component's actual path data
2. Scales to specified dimensions
3. Positions at specified attachment points
4. **Computes junction curves** where components meet (tangent-matching Bézier interpolation)
5. Outputs final `<path d="..."/>` with correct coordinates

The LLM selected components and described spatial relationships. The engine did all the Bézier math. The LLM never touched a control point.

**How many reference letters are needed:**

| Letters analysed | Components extracted | Coverage |
|---|---|---|
| 1 (F) | Stems, bars, serifs, joints | Straight letters only (E, I, L, T) |
| 3 (F, O, A) | + bowls, counters, diagonals | Most uppercase (~70%) |
| 5 (F, O, A, S, g) | + spine curves, descenders | Full alphabet feasible |

Each reference letter donates components the others don't have. F gives straight parts. O gives curves. A gives diagonals. S gives the reverse-curve spine. g gives descenders.

### 4.6 Spatial Grammar Reference

Common spatial patterns the LLM should use when writing intent. These are defaults — override with specific design knowledge when available.

**Face / head:**
```
Container: circular or oval, 60-80% of composition
Eyes: 2 circles, symmetric, upper third, each 3-5% of container
      Spacing between eyes ≈ 25-35% of container width
Nose/beak: on symmetry axis, between eyes and mouth
Mouth: on symmetry axis, lower third, width 20-40% of container
Eyebrow: above each eye, gap 20% of eye height, width 120-140% of eye
```

**Animal body (side view):**
```
Body: largest element, horizontal oval, 50-65% of canvas
Head: 20-30% of body, attached at front, may protrude
Legs: 2-4 elements, below body, vertical, each 5-10% of body
Tail: rear of body, 10-20% of body, extends beyond body
Eye: inside head, see face rules
```

**Radial / star / sun / flower:**
```
Center: 1 circle, 10-20% of composition
Petals/rays: N identical elements, evenly spaced at 360°/N
  Each: 5-15% of composition, extends from center outward
  REPEAT: N times, ROTATE: 360/N degrees
Partial: remove elements facing occluding object
```

**Font / letterform:**
```
Metrics: cap_height, x_height, ascender, descender, baseline
Stem: vertical, consistent width throughout font
Contrast: horizontal strokes thinner by consistent ratio
Serifs: identical shape at all terminals
Bowls: consistent curvature profile, overshoot at extremes
Counters: negative space proportional and consistent
```

**Stacked / layered:**
```
Back layer: largest, first in SVG document order
Each layer: offset by consistent (dx, dy), same or smaller
Offset direction: down-right most common
```

**Enclosed design (logo, badge):**
```
Container: geometric shape, outermost element
Content: inside with 10-15% padding all sides
Negative space: may carry meaning
```

### 4.7 Validation Loop

After the engine resolves intent into SVG coordinates, run the enrichment pipeline on the result. This is not reinforcement learning — it's a simple check.

```
LLM writes intent → Engine resolves → SVG output
                                          │
                              Engine enriches the output
                                          │
                              LLM reads enrichment of its own output
                                          │
                              Does the enrichment match the intent?
                                          │
                          ┌───────────────┴───────────────┐
                          │                               │
                        YES                              NO
                          │                               │
                        Done                    LLM adjusts intent
                                              (not coordinates —
                                               spatial description)
                                                      │
                                              Engine re-resolves
                                                      │
                                              Check again
```

What the LLM checks in the validation enrichment:

| Intent said | Enrichment should confirm |
|---|---|
| "eye inside head" | Containment: eye INSIDE head ✓ |
| "4% of head area" | Size: eye area = 3.8% of head ✓ (close enough) |
| "symmetric" | Symmetry score: 0.92 ✓ |
| "gap 20% of eye height" | Distance: eyebrow-to-eye = 0.19 of eye height ✓ |
| "LARGE/MEDIUM/SMALL tiers" | Size tiers show clear separation ✓ |
| "above eye" | Spatial: eyebrow bbox max-y < eye bbox min-y ✓ |

If something fails, the LLM adjusts the **intent description**, not the coordinates:

```
FAIL: "Containment: eye NOT inside head (1.2 units outside)"

LLM adjusts intent:
  Before: POSITION: inside head, upper-forward (75% x, 35% y)
  After:  POSITION: inside head, upper-forward (65% x, 40% y)
                                                  ↑ pulled inward

Engine re-resolves with adjusted percentages.
```

The LLM stays in its strength zone (spatial reasoning) and never enters its weakness zone (coordinate arithmetic). The loop typically converges in 1-2 iterations because the intent language is intuitive and the engine resolution is deterministic.

### 4.8 Creation Checklist

```
□ PLAN
  □ Canvas size chosen for complexity?
  □ Hierarchy sketched (what contains what)?
  □ Symmetry axis identified (if symmetric)?
  □ Component library available (if matching existing style)?

□ WRITE INTENT
  □ Every element has SHAPE, SIZE (% of parent), POSITION (relative)?
  □ Sizes expressed as percentages, never absolute coordinates?
  □ Positions expressed as regions (upper-forward, center, etc.)?
  □ Containment explicit (what's inside what)?
  □ Symmetry specified (mirror axis + which elements)?
  □ Depth order specified?
  □ Style specified (fill, stroke, classes)?
  □ Component references where available (font, icon set)?

□ ENGINE RESOLVES
  □ All elements have computed coordinates?
  □ Components scaled and positioned?
  □ Junctions computed (if component-based)?
  □ SVG output generated?

□ VALIDATE
  □ Run enrichment on output?
  □ Containment matches intent?
  □ Symmetry score acceptable (>0.85)?
  □ Size tiers match intent?
  □ All elements inside their parents?
  □ Silhouette matches expected shape?

□ ADJUST (if validation fails)
  □ Modify intent description (not coordinates)?
  □ Re-resolve and re-validate?
```

---

## PART 5: WHAT GOES WRONG AND WHY

### 5.1 Failure Taxonomy

Seven failure types discovered across 3 SVG analyses:

**TYPE 1: PREMATURE INTERPRETATION**
Naming the shape before measuring it. Auth0 — called them "shield pieces" before computing composite silhouette.
Fix: Complete ALL steps through cross-validation before any interpretation.

**TYPE 2: WRONG UNIT OF ANALYSIS**
Analyzing sub-paths individually when they should be analyzed as a group. Auth0 — ran convexity on each sub-path instead of the composite.
Fix: Always ask "individual or composite?" The tiling test answers this.

**TYPE 3: MISSING ANALYSIS SPACE**
Only analyzing positive (filled) space. Ignoring negative (empty) space. Auth0 — never looked at the star-shaped gaps.
Fix: Mandatory dual-space analysis for any multi-sub-path filled shape.

**TYPE 4: WRONG PIPELINE FOR SVG TYPE**
Using filled-shape metrics on stroke-based SVGs. Area-based circularity on an open stroke arc is meaningless.
Fix: Fill-attribute check is step zero. It gates everything else.

**TYPE 5: ACADEMIC TRANSFORM WORSHIP**
Including transforms because they sound impressive. BAS — "captures convexity at multiple scales!" In practice: told us "the cloud has concavities." Convex hull already said that.
Fix: Every transform must produce insight a simpler transform didn't already provide.

**TYPE 6: NOT TESTING BEFORE CUTTING**
Marking transforms as "useless" without computing them. Turning Function, Centroid Distance, CSS — all nearly cut, all turned out among the most useful.
Fix: Compute on 2–3 diverse SVGs before deciding.

**TYPE 7: FALSE SELF-CRITICISM**
Saying "I cheated" when the analysis actually worked. The sun arc: centroid distance PROVED circular before the image was ever rendered. The proof came first.
Fix: Track order of operations. If computation preceded visual confirmation, you didn't cheat — the pipeline worked. Trust the proof.

### 5.2 Case Studies

#### Case 1: Auth0 Shield + Star (FAILED)

**SVG:** 3 sub-paths, all CW winding. Right half (101.8 area), Left half (101.8, mirror), Bottom kite (54.5).
**Pipeline said:** "5-pointed star. 3-piece construction enables fractional fill."
**Actually:** Shield with 5-pointed star CUTOUT. Star is negative space.
**Root cause:** Only analyzed fills. Star vertices appear in filled polygons, so vertex analysis found pentagram angles. But the star is made of ABSENCE.
**Would have caught it:** Composite silhouette (T2.5) → shield. Negative space (T2.6) → star. Width profile span count > 1 → mandatory negative space analysis.

#### Case 2: Airbnb Bélo (SUCCEEDED)

**SVG:** 1 path, 2 sub-paths. Outer CW (large), Inner CCW (small) = hole.
**Pipeline said:** "Heart-like shape with smooth peak, non-uniform wall thickness (thin top=head, wide sides=arms, medium bottom=base), bilateral symmetry. Bélo."
**Why it worked:** Hole pattern → standard analysis. Wall thickness variation was the key discriminator.

#### Case 3: Weather "Partly Cloudy" (SUCCEEDED after course-correction)

**SVG:** 7 sub-paths, fill="none", stroke-width="2". 5 single-line rays + 1 open arc + 1 closed cloud.
**Key discoveries:**
- Stroke-based SVGs break filled-shape assumptions (ray casting, width profile meaningless)
- Turning function: arc = 227.6° (⅔ circle), cloud = 358.9° (closed)
- Centroid distance CV=0.090: circular (cheapest circle detector, works on open arcs)
- Directional coverage: rays span only 180° = sun on ONE SIDE ONLY = occlusion signal
- 5 identical lines at ~45° from shared center = rotation pattern = `<defs>` + `<use>` + `rotate()`

**Semantic reconstruction:** Raw 7-sub-path `<path>` → `<circle>` + `stroke-dasharray` for sun arc, `<line>` template with 5 `<use href="#ray" transform="rotate(N)">` for rays, `<path>` for cloud (irreducible organic shape). CSS variables for shared styling. The reconstructed SVG rendered identically to the original.

#### Case 4: Hugging Face Logo (SUCCEEDED — blind test)

**SVG:** 6 paths, stripped of colours/IDs.
**Enrichment said:** Bilateral symmetry about x-center (score 0.94). 2 circular shapes in upper region (eyes), arm shapes extending from body, cheek circles, smile curve.
**Pipeline said:** Correct identification from geometry alone. Every spatial relationship correct.
**Lesson:** Geometry accuracy is near-perfect for distinctive shapes with clear structural features.

#### Case 5: Apache Flink Logo (SUCCEEDED — geometry; brand needed colour)

**SVG:** 1 path, 4 sub-paths, stripped of colours.
**Pipeline said:** Pill body, border, circle, chevron arrow. Spatial relationships correct. Semantic interpretation correct (streaming/flow concept). Could NOT name the brand.
**Lesson:** Shape classification strong, but brand identification often requires colour as a disambiguating signal. Include colour metadata in enrichment.

---

## PART 6: EXECUTION CHECKLISTS

### 6.1 Analysis Checklist (run on every SVG)

```
□ PARSE (Layer 0)
  □ Expanded all commands?
  □ Resolved relative → absolute?
  □ Extracted sub-paths?
  □ Checked fill attribute (stroke vs fill)?
  □ Adaptive Bézier sampling (more points at high curvature)?

□ STRUCTURAL TRIAGE
  □ Classified each sub-path (line/curve/closed/open)?
  □ Measured all bboxes?
  □ Shape class auto-labeled (T1.23: circular/elliptical/rectangular/organic)?
  □ Size tiers assigned (LARGE/MEDIUM/SMALL)?
  □ Determined: single | hole | composite | multi-element?

□ CHEAP DISCRIMINATORS (every curved sub-path)
  □ Turning function?
  □ Centroid distance CV?
  □ Segment count and types?

□ PATTERN DETECTION (if multi-sub-path)
  □ Shared center test?
  □ Similarity test (Hu moment clustering)?
  □ Tiling test (if multiple filled, adjacent bboxes)?
  □ Angular spacing test (if similar sub-paths)?
  □ Containment matrix (winding number)?
  □ DBSCAN spatial clustering?

□ SPATIAL PICTURE
  □ ASCII grid — POSITIVE?
  □ ASCII grid — NEGATIVE (if multi-sub-path filled)?
  □ Width profile with span counts (if filled)?

□ TOPOLOGY (if 15+ elements)
  □ Connected component graph?
  □ Composite silhouette extraction (T3.20)?
  □ Visual stacking tree (T3.21)?
  □ Structural pattern report (T3.19)?

□ COMPOSITE (if multi-sub-path)
  □ Composite silhouette?
  □ Negative space description?
  □ Figure-ground report?

□ CROSS-VALIDATION (RESONANCE)
  □ Do features agree with each other?
  □ Does grid match numeric analysis?
  □ Does containment matrix agree with spatial clusters?
  □ Do symmetry pairs match shape similarity pairs?
  □ Any contradictions? → LOOP BACK

□ ENRICHMENT SUMMARY
  □ Assembled structured summary for LLM (§3.2 format)?

□ SYNTHESIS
  □ Overall shape + positive + negative + construction + identification?
```

### 6.2 Semantic Reconstruction Checklist

```
□ GEOMETRY DETECTION
  □ Circles/ellipses from cubic sequences (CV < 0.1)?
  □ Lines from cubics (collinear control points)?
  □ Repeated elements (same length/type cluster)?
  □ Shared centers / radial patterns?
  □ Shared styles?

□ LLM DECISIONS
  □ Object grouping (what belongs together)?
  □ Representation choice (circle vs arc vs cubics)?
  □ Template vs explicit (repeated elements)?
  □ Noise cleanup (round to clean values or keep)?
  □ What's irreducible (stays as <path>)?
  □ Foreground/background ordering?
  □ Semantic names and labels?

□ CODE GENERATION
  □ <defs> for reusable templates?
  □ <g> with transforms for groups?
  □ CSS classes + variables for shared styles?
  □ IDs and aria labels?
  □ Comments explaining construction logic?
  □ Component manifest CSV?
```

### 6.3 Pipeline Architecture

The engine and LLM communicate through a shared spatial vocabulary. The engine converts between spatial descriptions and coordinates in both directions.

```
═══════════════════ ANALYSIS FLOW (SVG → Understanding) ═══════════════════

  RAW SVG ──→ GEOMETRY ENGINE ──→ Enrichment Summary ──→ LLM READS
              (coordinates →                             (spatial
               spatial descriptions)                      reasoning)

═══════════════════ CREATION FLOW (Intent → SVG) ═════════════════════════

  LLM WRITES ──→ Spatial Intent ──→ GEOMETRY ENGINE ──→ NEW SVG
  (design                           (spatial descriptions
   reasoning)                         → coordinates)
                                          │
                                          ▼
                                    Engine enriches
                                    the output
                                          │
                                          ▼
                                    LLM validates:
                                    does enrichment
                                    match intent?
                                          │
                                    ┌─────┴─────┐
                                    YES         NO
                                    │           │
                                   Done    LLM adjusts
                                           intent (not
                                           coordinates)
                                              │
                                           Engine
                                           re-resolves
```

**The engine handles ALL coordinate math.** The LLM never reads or writes raw numbers. It reads enrichment summaries and writes spatial intent descriptions — same vocabulary, opposite direction.

Within the analysis flow, the LLM's reasoning is still a loop with checkpoints:

```
  Enrichment ──→ TRIAGE ──→ REASON ──→ CHECK ──→ INTERPRET ──→ OUTPUT
                   ▲                     │
                   │    contradiction     │
                   └─────────────────────┘
```

The CHECK step is where resonance happens. If measurements contradict each other or the grid, you DON'T proceed — you loop back to triage with a corrected understanding.

**This validation step is not optional. It's where failures get caught.**

### 6.4 Adaptive Selection (which transforms for which SVG type)

| SVG Type | Example | What To Run |
|---|---|---|
| Simple polygon | Triangle, hexagon | Parse → grid → vertices+angles → done |
| Complex single shape | Bélo | Parse → grid → width profile → curvature → wall thickness → symmetry |
| Composite tiling | Auth0 | Parse → DUAL grid → composite silhouette → width profile (spans!) → negative space → vertices → figure-ground |
| Stroke-based multi-element | Weather icon | Parse → fill=none → render strokes → turning → centroid dist → directional coverage → group → Layer 3 |
| Multi-element icon (≤15 paths) | Smiley, Hugging Face | Parse → grid → per-element properties → Layer 3 relationships |
| Complex icon (15–60 paths) | Animal, detailed logo | Parse → **full enrichment** (T1.23 shape class, T3.7 DBSCAN clusters, T3.1 containment matrix, T1.20 scored symmetry, T3.18 connected components, T3.19 heuristics, T3.20 silhouette, T3.21 stacking tree) → LLM reasons over enrichment summary |
| Illustrated (60–100+ paths) | Detailed animal art | Same as complex + T3.19 structure heuristics + report only top-N elements by area, summarize rest by tier |

---

## PART 7: ASSUMPTIONS & OPEN QUESTIONS

**A1. Grid resolution.** 32×32 for 24×24 viewBox icons. Hypothesis: `grid_size ≈ 1.5 × max(viewBox dimensions)`. For complex SVGs, silhouette extraction at 2× viewBox resolution.

**A2. Feature priority for LLM.** Current ranking:
1. Enrichment summary (if pre-computed)
2. ASCII grid POSITIVE + NEGATIVE
3. Composite silhouette description
4. Width profile with span counts
5. Vertex angles
6. Symmetry with mirror pairs
7. Basic properties + shape class
8. Curvature summary
9. Wall thickness

**A3. Negative space flood-fill.** Start from interior seed points (not border-touching cells). Border-touching empty space is "outside" not "negative space."

**A4. Morphological close threshold.** Gaps < 2% of bbox dimension → bridge. Larger → keep separate.

**A5. Token budget.** Full dual-space analysis ~3000 tokens. Enrichment summary ~1,200 tokens for 15 elements (compressible to ~800). For 60+ elements, report top 10 by area, summarize rest by tier. Total enrichment is a small fraction of context but eliminates 80% of measurement work.

**A6. Colour as semantic signal.** Shape classification from geometry alone is strong for distinctive shapes. Brand identification often requires colour. Include colours as metadata. `#000000` = neutral/icon. Named brand colors = identity hint.

**A7. Multi-resolution descriptions.** Untested. Theory: matches human description pattern. Worth A/B testing.

**A8. Medial axis.** Grid-based at 64×64 for organic filled shapes only. Skip for polygons and strokes.

**A9. Bézier context.** Classify curves as silhouette (outer boundary) vs structural (internal). Check composite silhouette boundary.

**A10. Stroke-based handling.** fill="none" → area-based metrics meaningless. Use turning + centroid distance + directional coverage. Affects ~50% of UI icon sets. The enrichment addendum assumes filled shapes — stroke-based complex SVGs are an untested gap.

**A11. Partial/occluded elements.** Open arc (turning < 360°) + directional coverage gap = partial hiding. Combine both signals.

**A12. Z-order from SVG document order.** SVG document order (path sequence) is the closest proxy for intended z-ordering. Later paths render on top. Preserve as metadata.

**A13. DBSCAN eps tuning.** eps = 8–12% of viewBox diagonal works for most icons. May need tuning for very dense or very sparse compositions. min_samples=2 (even pairs matter).

**A14. Computational cost.** Full enrichment pipeline for 60 elements: ~1 second total (50ms curve sampling + 100ms descriptors + 500ms containment matrix + 5ms DBSCAN + 10ms symmetry + 200ms silhouette).

**A15. Winding number robustness.** For complex/self-intersecting paths, test 5 points per element (centroid + 4 bbox midpoints) and majority-vote. Single-point test can give wrong results on concave shapes.

**A16. Texture/pattern fills.** SVGs with `<pattern>` fills need pattern-specific transforms (repetition period, tile shape). Not yet covered.

**A17. Text paths.** SVGs containing `<text>` along paths need OCR-like decomposition. Not yet covered.

**A18. VTracer integration (raster → SVG → enrichment).** VisionCortex's VTracer (`pip install vtracer`) converts raster images (PNG/JPG) to SVG. Pipeline: User drops PNG → VTracer vectorizes → VectorSight enriches → LLM analyzes. This would expand VectorSight from "SVG analysis" to "any image analysis" in one extra step. VTracer uses O(n) algorithms, handles high-resolution images, and produces clean stacked SVG output. Not needed for hackathon MVP but powerful demo extension.

**A19. Visual Stacking Tree vs flat element list.** Hypothesis: tree representation reduces LLM errors on containment and z-order questions. A flat list of 30 elements requires O(n²) pairwise reasoning. A tree with 4-5 root nodes and 5-8 children each is a scene graph the LLM can traverse. Needs A/B testing: same SVG, flat enrichment vs tree enrichment, measure accuracy on "what's inside what?" and "what's in front?" questions.

**A20. VisionCortex architectural parallel.** VisionCortex Impression builds: pixels → clusters → hierarchical binary tree → stacked vector paths. VectorSight builds: SVG coordinates → geometric features → enrichment hierarchy → spatial descriptions. Same pattern (raw data → semantic intermediate representation → consumer-readable output), different modality and consumer. VisionCortex processes pixels for algorithms. VectorSight processes vector geometry for LLMs. This parallel validates the architectural approach.

---

## PART 8: PIPELINE SUMMARY

**Total: 55 active transforms across 5 layers**
(Original 43 + 4 pattern detection + 2 shape descriptors + 5 topology/structure + 1 visual stacking tree)

| Layer | Count | Visibility | Purpose |
|---|---|---|---|
| 0 (Parsing) | 7 | INTERNAL | Clean raw SVG (T0.8 deferred) |
| 1 (Shape Analysis) | 21 active + 3 internal-only | INTERNAL | Features + shape class + EFD. 2 cut (BAS, TAR). T1.21 refined with VTracer signed-angle method. |
| 2 (Visualization) | 7 | LLM-FACING | Grids, profiles, silhouette, negative space, figure-ground |
| 3 (Relationships) | 21 | LLM-FACING | Inter-shape, tiling, gaps, patterns, components, heuristics, silhouette, visual stacking tree |
| 4 (Validation) | 5 | INTERNAL | QA + dual-space consistency + resonance |

**What's computed by geometry engine vs what needs LLM:**
- ~35 transforms computed by engine (deterministic, no LLM)
- ~10 transforms computed and summarized for LLM consumption
- ~10 require LLM reasoning (interpretation, grouping, naming)

**Tested status:**
- Proven (computed, changed outcome): T0.1-6, T1.1, T1.3, T1.4, T1.6, T1.8, T1.9-11, T1.14-15, T1.20-21, T2.1-7, T3.11-13
- Proven on blind tests (Hugging Face, Flink): T1.23, T3.7, T3.1, T1.18
- Internal/conditional: T1.5, T1.12, T1.13, T1.19, T1.24
- Never tested: T0.3, T0.7, T1.2, T1.16-17, T3.2-6, T3.8-10, T3.14-21, T4.1-5
- Cut: T1.7 (BAS), T1.22 (TAR), T0.8 (deferred)

### Implementation Libraries

| Layer | Library | Why |
|---|---|---|
| 0 (Parse) | `svgpathtools` | SVG path parsing, Bézier evaluation, `path.point(t)` |
| 0 (Parse) | `lxml` | Full SVG DOM: transforms, groups, styles |
| 1 (Descriptors) | `numpy` | Area (shoelace), centroid, PCA orientation |
| 1 (Descriptors) | `scipy.spatial` | ConvexHull, distance matrices |
| 2 (Containment) | Custom | Winding number on sampled contours |
| 3 (Clustering) | `scikit-learn` | DBSCAN for spatial and shape-similarity clustering |
| 4 (Structure) | `numpy` | Symmetry testing via reflection + matching |
| 3 (Components) | `networkx` | Connected component detection on spatial graph |
| 3 (Silhouette) | `scikit-image` | Rasterization, morphological ops, contour extraction |

---

## PART 9: SELF-IMPROVEMENT PROTOCOL

### 9.1 This Guide Evolves

This document is not static. It improves every time the system fails on an SVG. Each failure is a diagnostic signal pointing to a specific gap — a missing transform, a wrong threshold, an unclear instruction.

The pattern follows established self-improving AI techniques (Self-Refine, Claude Reflect, OpenAI Self-Evolving Agents) but applied to a domain-specific technical guide with measurable accuracy on a fixed test suite.

### 9.2 The Feedback Loop

```
SVG → Enrichment → LLM reasons using this guide → Answer
                                                     ↓
                                           Compare to ground truth
                                                     ↓
                                               ✅ Correct → log success
                                               ❌ Wrong → diagnose
                                                     ↓
                                          ┌──────────────────────┐
                                          │ 1. Enrichment gap?   │
                                          │    → fix transform   │
                                          │                      │
                                          │ 2. Data present but  │
                                          │    LLM missed it?    │
                                          │    → reorder/reword  │
                                          │    enrichment format │
                                          │                      │
                                          │ 3. Guide unclear?    │
                                          │    → rewrite section │
                                          │                      │
                                          │ 4. Threshold wrong?  │
                                          │    → adjust values   │
                                          └──────────────────────┘
                                                     ↓
                                            Patch guide → re-run → verify
                                                     ↓
                                            Run full test suite → no regressions?
                                                     ↓
                                               New guide version
```

### 9.3 Failure Log Format

Every failure generates a structured log entry:

```json
{
  "svg": "auth0-shield.svg",
  "guide_version": "v4",
  "enrichment_version": "2.1",
  "llm_answer": "5-pointed star",
  "correct_answer": "shield with star cutout",
  "root_cause": "missing_analysis_space",
  "root_cause_detail": "Only analyzed positive space. Star is negative space between filled polygons.",
  "missing_signal": "negative space analysis, composite silhouette",
  "transform_affected": "T2.6, T2.7",
  "fix_applied": "Made dual-space analysis mandatory for multi-sub-path filled shapes",
  "guide_section_updated": "Part 1 §1.6 (Dual-Space Analysis), Part 6 §6.1 (checklist)",
  "accuracy_before": 0.62,
  "accuracy_after": 0.78,
  "new_guide_version": "v5"
}
```

### 9.4 Root Cause Categories

Seven root causes identified so far (from Part 5). Each maps to a specific fix type:

| Root Cause | What Failed | Fix Type | Example |
|---|---|---|---|
| **Enrichment gap** | Transform doesn't compute needed data | Add/modify transform | T1.11 missing concavity positions → heart/leaf confusion |
| **Wrong pipeline** | Used filled-shape metrics on stroke SVG | Gate logic in guide | Fill-attribute check made Step 0 |
| **Missing analysis space** | Only analyzed positive space | Mandatory dual-space | Auth0 star was negative space |
| **Wrong unit of analysis** | Analyzed sub-paths individually instead of composite | Tiling test in triage | Auth0 3 pieces = 1 shield |
| **Premature interpretation** | Named shape before measuring | Step ordering in guide | Forced measurement before interpretation |
| **Guide ambiguity** | Instruction was unclear or missing | Rewrite guide section | Added explicit wording, examples |
| **Threshold error** | Shape class boundary wrong | Adjust numeric threshold | Circularity cutoff 0.85 → 0.80 |

### 9.5 Version Tracking

Each guide version records:

```
=== GUIDE CHANGELOG ===

v1 (initial): 34 transforms, baseline
v2: +T1.3 turning function, +T1.4 centroid distance (weather icon taught us)
v3: +dual-space analysis, +composite silhouette (Auth0 failure taught us)
v4: +T1.23 shape class, +T3.7 DBSCAN, +T3.18-20 topology (Hugging Face blind test)
v5: +T3.21 visual stacking tree, refined T1.21 corner detection (VisionCortex research)
...
```

### 9.6 Semi-Automated Diagnosis

The diagnosis step can use an LLM:

```
Prompt to Claude:
  "Here's the VectorSight enrichment for this SVG:
   [enrichment]

   The model answered: [wrong answer]
   The correct answer is: [correct answer]

   What specific data was missing from the enrichment
   that would have helped distinguish the correct answer?
   What guide section should be updated?"

Claude responds:
  "The enrichment reports 2 concavities but not their
   spatial positions. Heart has a top-center cleft,
   leaf has a base concavity. Adding position data to
   T1.11 would disambiguate. Update Part 2, T1.11
   and Part 6 checklist."

Human reviews → approves → guide patched → new version
```

The human stays in the loop for approval. The LLM handles the analysis. This is the same pattern as OpenAI's "metaprompt agent" that analyzes failures and suggests prompt refinements, but applied to a geometry-specific technical guide.

### 9.7 Improvement Curve as Demo

The accuracy-over-versions curve is itself a powerful demonstration:

```
Accuracy
  95% │                                          ●─── v7
  90% │                                    ●──── v6
  85% │                              ●──── v5
  80% │                        ●──── v4
  75% │                  ●──── v3
  70% │            ●──── v2
  65% │      ●──── v1
  60% │●───
      └──────────────────────────────────────────
       v1    v2    v3    v4    v5    v6    v7
       34T   36T   40T   43T   48T   54T   55T
       transforms
```

Each jump corresponds to a specific failure that was diagnosed and fixed. The curve proves the system learns — not through training, but through guide refinement driven by measurable failures.

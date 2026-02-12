# VectorSight Pipeline — How It Works

Hey! Here's the simple version of what our pipeline does.

## The Problem

LLMs can't *see*. When you give an LLM an SVG file, it just sees raw XML code — a bunch of `<path d="M 10,20 L 30,40 ..."/>` tags. It has no idea what that shape looks like, where things are relative to each other, or what the overall image depicts. Ask it "what animal is this?" and it's basically guessing from tag names and color values.

## Our Solution: Don't Make the LLM Guess — Tell It

We built a **geometry engine** that reads the SVG code, runs **62 math transforms** on it, and produces a structured text description of the spatial layout. Then we hand **both** the SVG and this enrichment text to the LLM. Now it actually *understands* the image.

No fine-tuning. No RAG. No special tools. Just better input data.

## The Pipeline (5 Layers)

Think of it like an assembly line. The SVG goes in one end, and at each layer, more information gets computed and attached.

```
SVG Code
  │
  ▼
┌─────────────────────────────────────────────┐
│  LAYER 0 — Parse                            │
│  "What shapes are in this SVG?"             │
│                                             │
│  Reads the raw SVG tags (path, circle,      │
│  rect, line, etc.) and converts them into   │
│  geometry objects with sampled points.       │
│  Bezier curves get sampled into point        │
│  sequences so we can do math on them.       │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  LAYER 1 — Shape Analysis (per element)     │
│  "What does each shape look like?"          │
│                                             │
│  For EACH element, computes:                │
│  - Bounding box, centroid, area             │
│  - Circularity (1.0 = perfect circle)       │
│  - Convexity (1.0 = no dents)              │
│  - Aspect ratio (wide vs tall)              │
│  - Shape class (circular, triangular,       │
│    rectangular, organic, linear...)         │
│  - Fourier descriptors (frequency-based     │
│    shape fingerprint)                       │
│  - Symmetry detection                       │
│  - Corner detection                         │
│  - Turning function analysis                │
│  - Color analysis                           │
│  - ...22 transforms total                   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  LAYER 2 — Visualization                    │
│  "What does the overall picture look like?" │
│                                             │
│  - Rasterizes shapes onto grids             │
│  - Builds ASCII art of the composite image  │
│  - Renders Unicode Braille grids (high-res) │
│  - Figures out figure vs ground             │
│  - Creates multi-resolution views           │
│  - Region density mapping                   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  LAYER 3 — Relationships (between elements) │
│  "How do shapes relate to each other?"      │
│                                             │
│  - Containment: "E2 is inside E5"           │
│  - Overlap: "E1 and E3 overlap (IoU=0.38)"  │
│  - Symmetry pairs: "E3 mirrors E4"          │
│  - Concentric groups: shapes sharing a      │
│    center (like an eye = circle in circle)  │
│  - Spatial clustering (DBSCAN): which       │
│    elements form visual groups              │
│  - Alignment detection                      │
│  - Repetition & pattern detection           │
│  - Nearest neighbors & distances            │
│  - Stacking/z-order                         │
│  - ...21 transforms total                   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  LAYER 4 — Validation                       │
│  "Does everything check out?"               │
│                                             │
│  Cross-checks between layers for            │
│  consistency. Catches edge cases.           │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  SPATIAL INTERPRETER                        │
│  "What's the big picture?"                  │
│                                             │
│  Synthesizes all 62 transforms into:        │
│  - Silhouette description (overall shape)   │
│  - Skeleton (topological backbone)          │
│  - Radial profile (protrusions & valleys)   │
│  - Mass distribution (where the "weight" is)│
│  - Focal elements (eyes, buttons, etc.)     │
│  - Scene decomposition (clusters)           │
│  - Shape narrative (natural-language         │
│    contour walk)                            │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  ENRICHMENT FORMATTER                       │
│  "Package it all for the LLM"              │
│                                             │
│  Takes all computed data and formats it     │
│  into structured plain text that the LLM    │
│  can read. Includes:                        │
│  - Spatial interpretation summary           │
│  - Shape narrative                          │
│  - Visual pyramid (ASCII art at multiple    │
│    zoom levels)                             │
│  - Step-by-step reconstruction guide        │
│  - Key elements with measurements           │
│  - Containment tree                         │
│  - Learned patterns from past sessions      │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
           ENRICHMENT TEXT
         (~1900 words, plain text)
```

## What the Output Actually Looks Like

Here's a real example — a simple smiley face SVG (4 elements: a circle, a mouth arc, two dot eyes):

```
=== VECTORSIGHT ENRICHMENT ===

ELEMENTS: 4 paths | CANVAS: 24×24 | TYPE: stroke-based

SPATIAL INTERPRETATION:
  LAYOUT SUMMARY: symmetric composition (strong bilateral)
  Composition: simple icon
  Orientation: strong bilateral symmetry; strong vertical symmetry (1.00)
  Silhouette: sparse, open composition; widest at top; roughly square proportions
  Shape topology: body mass with 1 narrow spike (right)
  Mass distribution: evenly distributed; vertically centered
  Focal elements:
    - E3: circular (circ=1.00) inside E2
    - E4: circular (circ=1.00) inside E2
  Reading hints:
    - High symmetry (1.00) — balanced/symmetric design
    - 3 circular elements.

SHAPE NARRATIVE:
  Contour: Starting from bottom-center: rises upper-right at upper (broad curved
  lobe), rises short upper-left at upper, extends left at upper (broad curved lobe),
  descends short down at upper, extends short right at upper.

── VISUAL PYRAMID ──

LEVEL 0 — ALL ELEMENTS (12×12):
............
...######...
..#......#..
.#........#.
.#.##..##.#.
.#..#...#.#.
.#........##
.#..#####.#.
.#........#.
..#......#..
...######...
.....#......

RECONSTRUCTION (build image mentally, step by step):
  1. E2 [LARGE] circular at center
  2. E1 [MEDIUM] triangular at center
  3. E3 [SMALL] circular at center
  4. E4 [SMALL] circular at right

KEY ELEMENTS (top 4 by area):
  E2: [LARGE] circular(closed_loop), bbox(2,2,22,22), at(12,12), circ=1.00
    svg: <circle cx="12" cy="12" r="10"/>
  E1: [MEDIUM] triangular(curved), bbox(8,14,16,15), at(12,14), circ=0.47
    svg: <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
  E3: [SMALL] circular(closed_loop), bbox(7,8,9,10), at(8,9), circ=1.00
    svg: <circle cx="8" cy="9" r="1"/>
  E4: [SMALL] circular(closed_loop), bbox(15,8,17,10), at(16,9), circ=1.00
    svg: <circle cx="16" cy="9" r="1"/>

SYMMETRY: vertical (score 1.00)
  Pairs: (E3↔E4)

CONTAINMENT TREE:
  E2 [LARGE] circular at center (3 children)
    E1 [MEDIUM] triangular at center
    E3 [SMALL] circular at center
    E4 [SMALL] circular at right

=== END ENRICHMENT ===
```

Look at that ASCII art grid — you can literally *see* the smiley face. The LLM gets this text, and now it knows:
- There's a big circle (E2) containing everything
- Two small identical circles (E3, E4) are symmetrically paired — those are eyes
- There's a curved arc below them (E1) — that's the mouth
- The whole thing has perfect vertical symmetry

Without our enrichment, the LLM just sees `<circle cx="8" cy="9" r="1"/>` and has to guess what that means spatially.

## How the LLM Receives It

When a user asks something like *"What does this SVG look like?"*, here's what the LLM actually gets in its prompt:

```
=== SVG CODE ===
<svg viewBox="0 0 24 24">
  <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor"/>
  <path d="M8 14s1.5 2 4 2 4-2 4-2" .../>
  <circle cx="8" cy="9" r="1"/>
  <circle cx="16" cy="9" r="1"/>
</svg>

=== SPATIAL ENRICHMENT (auto-computed by geometry engine) ===
[...the entire enrichment text above...]
```

The LLM also gets a detailed system prompt explaining how to read each section of the enrichment — what circularity means, how to interpret the ASCII grid, how to trace the contour walk, etc.

## The Key Insight

The enrichment isn't just numbers — it's structured at multiple levels of abstraction:

| Level | What it gives the LLM | Example |
|-------|----------------------|---------|
| **Raw measurements** | Exact coordinates, areas, ratios | `bbox(2,2,22,22), circ=1.00` |
| **Shape classification** | What each element IS | `circular(closed_loop)` |
| **Relationships** | How elements relate | `E3↔E4 symmetric pair` |
| **Visual grids** | ASCII "pictures" it can see | The 12x12 smiley grid |
| **Natural language** | Human-readable shape walk | `"broad curved lobe at upper"` |
| **High-level synthesis** | The "big picture" | `"symmetric composition, simple icon"` |

The LLM gets to choose which level of detail to use depending on the question. Simple question? Read the layout summary. Need precision? Check the bbox coordinates.

## For Complex SVGs

The smiley was simple (4 elements). For complex illustrations like the Apache Flink logo (67 elements), the enrichment includes:
- Scene decomposition into spatial clusters
- Concentric group analysis (layered features like eyes)
- Detailed contour walk of the primary boundary
- Multiple ASCII grids at different zoom levels (composite, boundary only, interior features, per-group zooms)
- Step-by-step mental reconstruction guide
- Color palette analysis
- Overlap and containment hierarchies

All of this stays under ~1900 words — designed to fit well within LLM context windows without wasting tokens.

## One Line Summary

**We do the geometry math so the LLM doesn't have to.**

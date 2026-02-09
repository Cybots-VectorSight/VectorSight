# Plan: Deep Spatial Correlation Reference for VectorSight Agents

## What This Is

A reference document that describes what each VectorSight transform's output **means in terms of visual perception** — based on established research, not hardcoded thresholds. Written for agents doing reverse engineering: given spatial measurements, what does the combination suggest?

The reference describes **directional relationships and research findings**, not cutoff values. The agent decides what specific numbers mean in context — only it has the full picture while analyzing.

## Why No Hardcoded Values

Research describes correlations qualitatively:
- "Higher circularity correlates with rounder perceived shape" — NOT "0.9 = circle"
- "Concavities carry more information than convexities" — NOT "solidity < 0.5 = text"
- "Bilateral symmetry with vertical axis → strongest biological recognition" — NOT "score > 0.75 = front-facing"

The exact meaning depends on what ALL other transforms say simultaneously. Circularity of 0.85 could be a head, a fruit, or a button — the agent synthesizes from context.

## Output File

`backend/app/learning/spatial_reference.md` — static reference, never edited by the agent.

---

## Structure

Organized by **reverse-engineering questions**, not by transform ID.

### Section 1: "What does the shape of this element suggest?"

**Transforms involved:** T1.15 Circularity, T1.11 Convex Hull Ratio, T1.16 Rectangularity, T1.21 Corner Detection, T1.23 Shape Class, T1.1 Curvature Profile

**Research to describe (qualitative relationships only):**
- Circularity measures how close to a perfect circle. Higher → rounder. Lower → more irregular or elongated. The relationship is continuous, not categorical
- Convex hull ratio (solidity) measures concavity. As it decreases, the shape has deeper indentations — branching, fingers, star points, internal gaps. Concavity itself is meaningful
- Corner count relates to whether a shape is polygonal (many defined vertices) vs. organic (smooth, few inflection points)
- Curvature profile: smooth continuous variation suggests organic/natural origin. Constant zero curvature with sharp transitions suggests manufactured/geometric. Uniform constant curvature suggests mechanical (arcs, circles). High-frequency oscillation suggests decorative/ornamental

**Key research:**
- **Attneave (1954)**: information along contours concentrates at curvature extrema. These points (corners, tips, sharp bends) are the most semantically important — connecting only these points preserves recognizability
- **Koenderink & van Doorn (1992)**: shape index classifies local curvature into perceptual categories (cup, trough, saddle, ridge, dome)

### Section 2: "What kind of subject might this be?"

**Transforms involved:** T1.20 Symmetry, T1.14 Aspect Ratio, T1.17 Eccentricity, T2.5 Composite Silhouette, T1.9 Width Profile

**Research to describe:**
- Bilateral symmetry with vertical axis correlates strongly with biological organisms — evolutionary pressure makes animals vertically symmetric. Also correlates with designed functional objects (vehicles, buildings)
- Rotational symmetry of various orders correlates with different natural/designed categories (flowers, mechanical parts, decorative patterns)
- Aspect ratio indicates proportionality — equidimensional vs elongated — but what that means depends entirely on other features. A tall narrow shape could be a standing figure OR a pencil OR a divider
- Width profile with multiple spans indicates internal negative space — gaps, holes, cutouts that are themselves features
- The composite silhouette (outer boundary shape) is "what you'd see if you squinted" — curved outlines lean organic, rectangular lean geometric, but this is a tendency not a rule

**Key research:**
- **Biederman (1987)** Recognition-by-Components: humans decompose objects into ~36 geometric primitives (geons). Each defined by cross-section curvature, symmetry, and size change along axis. Means: combinations of geometric descriptors map to recognizable object parts — but the mapping is learned, not a lookup table
- **Elongation perception bias** (PMC, 2023): humans systematically overestimate the area of elongated shapes — elongation is perceptually salient

### Section 3: "What does the spatial arrangement reveal?"

**Transforms involved:** T1.20 Symmetry axis, T2.2 Region Map, T2.7 Figure-Ground, T3.5 Relative Position

**Research to describe:**
- The direction of a symmetry axis, when present, relates to how the subject is oriented. Vertical axis is processed fastest perceptually (Gestalt). But moderate symmetry can mean many things — symmetric features in an asymmetric composition
- Mass distribution across the canvas (region map density) reveals where visual weight sits. Asymmetric mass distribution correlates with directional facing, but the agent must consider what the mass IS
- Figure-ground assignment: perceptual research shows smaller, more enclosed, more convex, more symmetric regions are perceived as "figure" (foreground). But clever designs intentionally invert this

**Key research:**
- **Wagemans et al. (2012)** "A Century of Gestalt Psychology": proximity, similarity, closure, good continuation, symmetry, common region — all affect how elements are grouped and interpreted. These principles describe tendencies in human perception that the agent should understand
- **Gestalt figure-ground**: vertical symmetry, convexity, smallness, enclosure all bias toward "figure" perception

### Section 4: "How should complexity be interpreted?"

**Transforms involved:** Element count, T3.1 Containment depth, T3.21 Visual Stacking Tree, T3.18 Connected Component Graph, T3.7 Grouping, T1.1 Curvature variation

**Research to describe:**
- Complexity is multi-dimensional — element count alone is misleading. A few deeply nested organic elements with high curvature variation can be perceptually more complex than many repeated simple shapes
- Nesting depth indicates hierarchical structure. Deeper nesting correlates with more detailed, layered compositions — but the type of elements matters more than the depth number
- Stacking/layering of shapes is itself an illustration technique — overlapping shapes build up surfaces, shading, color regions
- Clustering reveals whether elements form coherent groups or are scattered

**Key research:**
- **Forsythe et al. (2011)**: fractal dimension and visual complexity together explain 42% of beauty variance in art. Moderate complexity tends to be aesthetically preferred over extreme simplicity or extreme complexity
- **Cutting & Garvin (1987)**: curvature variation along contours correlates with perceived complexity. High variation = complex organic form

### Section 5: "What do spatial relationships between elements mean?"

**Transforms involved:** T3.1–T3.6 (Containment, Distance, Alignment, Relative Size, Relative Position, Overlap), T3.15 Shared Center, T3.16 Angular Spacing, T3.17 Occlusion

**Research to describe:**
- Containment (A inside B): B is container/context for A. Tight containment vs loose containment are perceptually distinct categories (cross-linguistic research on spatial cognition)
- Distance between elements: proximity is the strongest Gestalt grouping cue. Elements close to each other are perceived as belonging together. But "close" is relative to their own size
- Touching/overlapping elements (zero distance): form continuous visual surfaces. Common in illustrations using stacked shapes for shading or color regions
- Shared center: multiple elements arranged around a common center suggests intentional radial/rotational composition
- Angular spacing near clean multiples: suggests tool-generated precision. Irregular spacing suggests hand-placed
- Partial visibility (open arcs, truncated shapes): occlusion cue — something is hidden behind something else

**Key research:**
- **Choi & Bowerman (1991)**: containment is one of the earliest spatial categories learned (by 3 months). Tight-fit vs. loose-fit containment are perceptually primary
- **Gestalt proximity**: the dominant grouping principle. Overrides similarity, color, and other cues when distance is sufficiently close

### Section 6: "What does positive/negative space reveal?"

**Transforms involved:** T2.1 ASCII Grid, T2.6 Negative Space, T2.7 Figure-Ground, fill percentage

**Research to describe:**
- The ASCII grid is a low-resolution rasterization — read it as a picture, not as data
- Negative space (gaps, holes, empty regions) can be features themselves, not just absence. Many iconic designs use negative space to encode meaning (FedEx arrow, Auth0 star)
- Figure-ground perception determines which regions carry the intended meaning. This is NOT always the filled regions
- Fill ratio is continuous and context-dependent. It also reflects rendering technique (stroke vs fill) — a stroke-rendered circle has low fill percentage but is visually solid

**Key research:**
- Figure-ground has three perceptual types: stable (clear distinction), reversible (both interpretations valid — Rubin's vase), ambiguous (no clear assignment — tessellations)
- **Smaller, enclosed, convex, symmetric** regions bias toward figure perception. Larger, surrounding regions bias toward ground

### Section 7: "What patterns and repetitions exist?"

**Transforms involved:** T3.9 Repetition, T3.14 Repeated Element Detection, T3.11 Tiling Classification, T3.8 Multi-shape Symmetry, T3.19 Structural Pattern Report

**Research to describe:**
- Repeated elements with similar descriptors suggest intentional pattern. The Gestalt principle of similarity groups them perceptually
- Tiling vs nesting vs separation describes fundamentally different compositional strategies
- Mirror pairs across a composition suggest bilateral design. Rotational arrangement suggests radial design
- The spatial arrangement of similar sub-shapes is itself a feature — regular grid = texture, irregular = natural distribution, unique shapes in structured arrangement = diagram

**Key research:**
- **Shape Clustering (Pattern Recognition, 2012)**: shapes with similar multidimensional descriptors that are spatially proximate are perceived as groups
- **Gestalt similarity**: matching properties across elements create grouping

### Section 8: "When features seem to contradict"

**Transforms involved:** T4.5 Dual-Space Check, T1.18 Hu Moments, resonance rules

**Research to describe (these are from the VectorSight guide itself, proven in practice):**
- Positive and negative space should tell compatible stories. When they disagree, investigate which interpretation is correct
- Elements with very similar Hu moment distances are likely the same template shape repeated/transformed
- Width profile showing multiple spans on shapes classified as "separate" suggests they actually tile together as composite
- Grid showing connected fill contradicting "multiple elements" analysis suggests composite treatment
- Values near clean angle multiples likely reflect tool precision, not meaningful variation
- Nearly identical sub-path features indicate repeated elements — check for transform relationships

### Section 9: "What curvature reveals beyond shape"

**Cross-cutting research findings:**
- **Attneave (1954)**: information is concentrated at curvature maxima. His "sleeping cat" experiment — connecting only curvature peaks with straight lines preserves recognizability. Implication: the most semantically important points are corners, tips, and sharp bends
- **Feldman & Singh (2005)**: formalized Attneave using information theory. Concavities carry MORE information than convexities of equal magnitude, because concavities are less expected. Part boundaries are perceived at negative curvature minima — this is where shapes are decomposed into parts
- **Bar & Neta (2007)**: angular shapes activate the amygdala (fMRI). Sharp angles are processed as threat signals. This is automatic and pre-conscious. Curved shapes are perceived as safer/calmer
- **Cross-modal associations (PLOS One, 2015)**: curvature carries implicit meaning beyond vision — curved shapes associate with softness, sweetness, calm. Angular shapes with hardness, bitterness, aggression. This is cross-cultural
- **Blum (1967)** Medial Axis: the skeleton of a shape reveals its topology. Branching skeleton = articulated parts (limbs, fingers). Single axis = simple elongated form. Radial axes = central body with extensions

### Section 10: "Common traps — what NOT to assume"

From the VectorSight guide (proven through analysis) + research:
- **Figure-ground inversion**: the filled regions may not be the intended subject. Some designs encode meaning in the negative space
- **Sub-path ≠ separate object**: SVG sub-paths are usually construction pieces of ONE design, not separate objects
- **Construction vs. visual**: multiple path commands that build one visual shape. The code structure doesn't reflect the visual structure
- **Repetition ≠ complexity**: many identical repeated shapes inflate element count without adding perceptual complexity
- **Single-feature interpretation**: no single transform output determines what something is. Always synthesize from multiple features
- **Rendering technique ≠ visual property**: stroke vs fill, layered shading vs flat color — these are construction methods, not visual characteristics

---

## Research Sources

### Foundational Perception
- **Attneave (1954)** "Some Informational Aspects of Visual Perception" — info at curvature extrema
- **Biederman (1987)** "Recognition-by-Components" — objects = combinations of ~36 geons
- **Feldman & Singh (2005)** "Information Along Contours" — concavities > convexities for information
- **Blum (1967)** "Medial Axis Transform" — skeleton reveals topology
- **Wagemans et al. (2012)** "A Century of Gestalt Psychology" — definitive modern Gestalt review

### Shape Analysis
- **Bribiesca (2009)** "State of the Art of Compactness and Circularity Measures"
- **Koenderink & van Doorn (1992)** "Shape Index" — local curvature classification
- **Choi & Bowerman (1991)** — infant spatial categorization (containment as primary category)

### Curvature & Perception
- **Bar & Neta (2007)** angular shapes activate amygdala (fMRI threat detection)
- **Bertamini et al. (2016)** "Do Observers Like Curvature or Dislike Angularity?"
- **PLOS One (2015)** curved/angular implicit cross-modal associations
- **Scientific Reports (2015)** convexities recognized faster than concavities

### Symmetry
- **PNAS (2005)** symmetry activates extrastriate visual cortex
- **Vision Research** symmetry perception stronger for biological shapes

### Complexity & Aesthetics
- **Forsythe et al. (2011)** fractal dimension + visual complexity explain 42% of beauty variance
- **Cutting & Garvin (1987)** curvature variation and perceived complexity

### SVG & LLM
- **VGBench (EMNLP 2024)** LLMs struggle with raw SVG geometry
- **VDLM** structured primitive descriptions improve LLM SVG reasoning
- **LLM4SVG (CVPR 2025)** tokenized SVG representations cause hallucinations
- **SVGenius (2025)** SVG understanding evaluation across 24 domains

---

## Implementation Steps

### Step 1: Deep research each section
For each of the 10 sections, do thorough web research focused on the specific transforms involved. Find what research says about the **qualitative relationship** between each geometric measurement and visual perception. No fabricated thresholds — only describe directions, tendencies, and research findings.

### Step 2: Write `backend/app/learning/spatial_reference.md`
All 10 sections. Each section:
- Lists which transforms are involved
- Describes what the measurement relates to perceptually (directional only)
- Cites specific research findings
- Notes how features interact with each other in that section
- NO hardcoded value ranges or classification tables

### Step 3: Add reader function to `tags.py`
```python
_REFERENCE_PATH = Path(__file__).parent / "spatial_reference.md"

def get_spatial_reference() -> str:
    """Read the static spatial correlation reference."""
```

### Step 4: Inject into reflection prompts
Update `self_reflect.py` to include both:
```
SPATIAL REFERENCE (research-based correlations):
{spatial_reference}

PRIOR KNOWLEDGE (learned from experience):
{knowledge}
```

### Step 5: Compress for token budget
The full reference may be too long for every reflection call. Create a compressed prompt-friendly version (~2000 tokens). Keep full version as file the agent can reference.

### Step 6: Consider injecting into main enrichment
`enrichment_formatter.py` could append relevant correlation context based on which features are present — qualitative hints, not thresholds.

### Step 7: Test
- All tests pass
- Live reflection with reference vs without — does the agent make better tag assignments?
- Token usage stays within budget (~4000 tokens total for reflection prompt)

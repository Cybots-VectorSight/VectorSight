# Spatial Perception Reference

> How to interpret geometry measurements visually.
> You see enrichment data — this reference helps you understand what it means.
> No single measurement determines identity. Always synthesize across features.

---

## How to Read the Enrichment

**Priority order** — Read these sections first, they carry the most signal:
1. **STACKING TREE** — The scene graph. Read it as structure: outer element contains inner elements (e.g., frame contains icon, face contains features). The tree structure IS the design's architecture.
2. **CONTAINMENT** — What's inside what. This tells you part-of relationships directly.
3. **SYMMETRY + pairs** — Orientation, subject type, paired features (windows, wheels, eyes, wings).
4. **CONCENTRIC groups** — Eyes, targets, wheels, buttons. Position relative to containing element = identity.
5. **PER-ELEMENT (top 15)** — Shape features of the biggest elements. Focus on LARGE tier first.
6. **SPATIAL INTERPRETATION** — The pipeline's own best guess. Start here for a quick read.

**Skim or skip** — These are supplementary:
- ALIGNMENTS — Only meaningful when groups have 4+ elements in a line.
- OVERLAPS — High IoU (>0.5) = stacked shading layers. Low IoU = barely touching. Mid-range IoU = partial overlap (occlusion).
- NEAREST NEIGHBORS — When "most elements touching" appears, it means a layered illustration style. Individual distances matter only when elements are separated.
- REPEATED ELEMENTS — Only useful when shapes are actually geometrically similar, not just sharing a shape_class.

**Reading Braille grids** — The enrichment uses Unicode Braille characters (⠀ through ⣿) for high-resolution text rendering. Each character encodes a 2-wide × 4-tall dot matrix (8 pixels per character), giving 8× more resolution than regular text art.

How to read them:
- Each dot (raised position) in the Braille character = a filled pixel. Empty positions = empty pixels.
- The SILHOUETTE grid shows the composite shape of all elements merged. Read the overall outline — where it's widest/narrowest, what the silhouette looks like.
- ELEMENT SHAPES show individual top elements rendered in isolation within their bounding box. These reveal each element's actual shape.
- **CONCENTRIC GROUP SHAPES** are the most informative visual section. They render concentric groups (elements sharing a center) as combined Braille mini-grids. This shows what each group IS — an eye (ellipse + circle + dot), an acorn (circle + cap shape), a wheel (circle + spokes), a button (circle + inner ring), etc. Examine these carefully.

Tips for reading Braille grids:
- Look for the overall silhouette outline first, then identify internal features.
- Circular shapes appear as dot clusters forming arcs.
- The contrast between filled and empty regions shows the shape boundary.
- Compare multiple group grids to identify which features are similar/different.

**Reconstructing structure from centroids** — Each element's centroid is its center of mass. To see the overall structure:
- Plot the LARGE elements' centroids mentally — they form the skeleton of the composition.
- Elements whose centroids are above the midline = upper features, crown, peaks.
- Elements whose centroids cluster at the sides = lateral extensions, wings, arms, flanges.
- Elements with centroids below = base, ground contact, foundation.
- Cluster centroids show where each structural region sits spatially.

---

## Shape Features (per-element data)

**shape_class** — The pipeline's own classification (circular, elliptical, rectangular, triangular, linear, organic). Use as a starting point, not a final answer. A "circular" element could be a head, eye, wheel, button, fruit, or dot — context decides.

**circularity** — How close to a perfect circle (1.0 = circle). Higher = rounder. Lower = more irregular, elongated, or spiky. Many different things share similar circularity, so it constrains but never determines identity.

**convexity** (convex hull ratio) — Measures concavity. Near 1.0 = fully convex, no indentations. As it drops: branching, fingers, star points, internal gaps appear. Key insight: concavities carry more perceptual information than convexities — they're where humans naturally decompose shapes into parts. Low convexity = the shape has distinct sub-parts.

**aspect_ratio** — Width-to-height proportionality. Near 1.0 = equidimensional. Elongated shapes are perceptually salient — humans overestimate their area. But aspect ratio alone says nothing about identity. A tall narrow shape with bilateral symmetry suggests different things than one without.

**corners** — More corners = more polygonal/geometric. Few or zero = organic/smooth. The corners themselves are the most informative points on any shape boundary — they carry the most identity information.

**GAPS flag** — When width profile shows multiple spans at a height level, the shape has internal holes or cutouts. These gaps are features, not just absence — windows in a building, eyes in a face, letters cut from a background, holes in a gear.

**turning_classification** — Identifies open arcs vs closed loops. An open arc covering less than 360 degrees means something is either incomplete or partially hidden. Combined with directional coverage, this reveals occlusion.

**size_tier** (LARGE/MEDIUM/SMALL) — Relative importance within the SVG. LARGE = structural elements (outlines, backgrounds, frames). MEDIUM = features (eyes, buttons, distinct parts). SMALL = details (highlights, dots, decorations). The tier hierarchy is itself a signal about what matters.

**Curvature patterns** (from the shape fingerprint):
- Smooth continuous variation → organic, natural origin
- Flat segments with sharp transitions → manufactured, geometric
- Uniform constant curvature → mechanical (arcs, gears, wheels)
- High-frequency oscillation → decorative, ornamental detail
- Angular shapes carry implicit threat/aggression associations; curved shapes carry safety/calm associations — this is cross-cultural and automatic

---

## Global Shape Descriptors (composite silhouette analysis)

These three features analyze the COMPOSITE silhouette (all elements merged) to provide text-based global shape identity signals. They are more reliable than the grid for understanding overall form.

**Skeleton (medial axis)** — The topological backbone of the shape. Key signals:
- Branch count = structural complexity. 0 branches = blob/circle. 1 main axis = elongated form. 2-3 branches = T-shape, Y-shape, animal profile. 4+ branches = star, cross, hand, complex figure.
- Branch directions tell you WHERE the shape extends. "Main vertical axis + upper-right branch + lower-left branch" suggests a diagonal composition or side-profile creature (head + tail).
- Junction count = articulation points. More junctions = more distinct body parts or structural regions.
- Two shapes with different skeletons are different things. Two shapes with similar skeletons may be the same thing.

**Radial profile** — Distance from centroid to boundary at 24 equally-spaced angles (every 15°, starting at 0°=right, counter-clockwise).
- Read it as a shape fingerprint: peaks correspond to protrusions (head, tail, wings, limbs, ears, roof peaks), valleys correspond to concavities (neck, waist, between legs, indentations).
- Peak angles tell you WHERE protrusions point. A peak at 0° (right) and 180° (left) = bilateral horizontal extensions (wings, arms). A peak at 90° (up) = upward extension (head, antenna, chimney).
- Valley angles tell you WHERE the shape narrows. A valley between two peaks = two adjacent body parts or structural regions.
- Approximately circular shapes have flat profiles. Highly articulated shapes have spiky profiles.

**Simplified contour path** — A simplified boundary trace as SVG M/L commands with integer coordinates.
- Mentally trace the path to visualize the outline. Direction changes between vertices mark corners, indentations, and protrusions.
- Fewer vertices after simplification = smoother, more regular shape. More vertices = complex, irregular boundary.
- This is the shape the viewer actually sees — the composite outline of all elements merged together.

---

## Symmetry & Orientation

**SYMMETRY axis + score** — The type and strength of symmetry.
- Strong vertical bilateral symmetry → organisms face forward; objects have primary sides (vehicles, buildings, tools)
- Strong horizontal bilateral → lying/resting orientation or side-profile views
- Rotational symmetry (n-fold) → flowers, mechanical parts, stars, mandalas, snowflakes, decorative patterns
- Moderate symmetry score with identifiable mirror pairs → paired features (windows, wheels, eyes, wings) in an otherwise asymmetric composition — possibly a three-quarter view or dynamic pose
- No symmetry → organic irregular forms, text, hand-drawn elements

**Symmetry pairs** — Which elements mirror each other. Mirror pairs across a vertical axis suggest front-facing bilateral design. The elements that sit ON the axis (not paired) are typically central features — nose, door, body midline.

**Orientation from mass distribution** — Where visual weight sits in the region map:
- Mass concentrated left or right → side-facing or directional composition
- Centered mass → front-facing or symmetric design
- Top-heavy → upright standing figure, tower, tree
- Bottom-heavy → grounded object, vehicle, landscape
- Offset mass → implied motion or directional tension

---

## Spatial Relationships

**CONTAINMENT** — A contains B means B is inside A. This immediately creates a structural hierarchy. Tight containment (B nearly fills A) versus loose containment (B is small within A) are perceptually distinct — tight = part-of, loose = feature-within.

**NEAREST NEIGHBORS** — Proximity is the strongest grouping cue. Elements close together are perceived as belonging together. "Close" is relative to the elements' own size. When distance is zero = touching/overlapping → continuous visual surface, stacked shading, or layered depth.

**OVERLAPS** — Partial overlap creates occlusion — the element rendered later (higher z-order) is perceived as in front. Full overlap of similar shapes usually means shading or color variation. The visual system automatically infers what's hidden behind an occluding shape.

**RELATIVE POSITIONS** — The spatial arrangement of parts reveals structure. "B is above A" combined with containment and size tells you about anatomy, composition, or scene layout.

**ALIGNMENTS** — Elements sharing vertical or horizontal center lines suggests intentional arrangement — aligned features, structured layout, or designed symmetry.

**CONCENTRIC groups** — Multiple elements sharing a center point → radial or concentric composition. Common for: eyes (iris + pupil + highlight), targets, wheels, flowers, decorative rosettes, buttons, dials.

**ANGULAR SPACING** — For radially-arranged elements: regular spacing near clean multiples (15, 30, 45, 90 degrees) = tool-generated precision. Irregular spacing = hand-placed or organic arrangement. This tells you about the type of artwork.

---

## Complexity & Structure

**Element count alone is misleading.** A few deeply nested organic elements with high curvature variation can be perceptually more complex than many repeated simple shapes. What matters: variety of shapes, hierarchical depth, and curvature variation — not raw count.

**STACKING TREE** — The containment hierarchy as a scene graph. Read it as: "container holds features and details" rather than "E1, E2, E3, E4 are separate things." The tree structure is more meaningful than a flat list.

**SPATIAL CLUSTERS** — How elements group spatially. One dominant cluster + small satellite clusters → main structure with extensions or accessories. Multiple equal clusters → scene with multiple focal points or repeated units. Isolated elements → disconnected details or background.

**VISUAL UNITS** (connected components) — How many independent visual groups exist. One component with internal structure = single illustration. Multiple disconnected components = scene, set, or diagram.

**STRUCTURE report** — Recurring geometric arrangements described factually. These are patterns, not identifications — a "large containing element + 2 symmetric small elements in upper quadrant + 1 element in lower center" is a structural description. YOU decide what it depicts.

**TOPOLOGY** — The overall compositional structure type. Combined with stacking tree, tells you whether you're looking at a single subject, a scene, or a pattern.

---

## Positive & Negative Space

**GRIDS** — Read these as pictures. The positive grid shows what's filled using half-block characters for higher resolution. Element mini-grids show individual top elements.

**FIGURE-GROUND** — Which regions carry the intended meaning.
- Smaller, more enclosed, more convex, more symmetric regions → perceived as "figure" (foreground)
- Larger surrounding regions → perceived as "ground"
- But some designs deliberately invert this — meaning lives in the GAPS, not the fills
- Three types: stable (clear), reversible (both valid — some logos exploit this), ambiguous (tessellations)

**The filled regions may NOT be the intended subject.** Some designs encode meaning in negative space. When you see the positive grid showing one thing but the negative grid revealing a recognizable shape in the gaps — both are probably intentional.

**Fill percentage** reflects rendering technique, not visual density. A stroke-rendered circle has low fill percentage but is visually perceived as a complete solid form. Don't confuse SVG construction method with what the viewer sees.

---

## Patterns & Repetition

**REPEATED ELEMENTS** — Elements with matching shape descriptors that appear multiple times. Repeated elements can be:
- Decorative (spots, scales, feathers, stars in a sky, rivets)
- Structural (teeth, fingers, petals, spokes, columns)
- Organizational (icons in a grid, items in a list, dashboard widgets)

The spatial arrangement of repeated elements is itself a feature:
- Regular grid → texture or pattern fill
- Irregular scatter → natural distribution
- Radial arrangement → flower, gear, star, mandala, dial
- Linear arrangement → border, fence, sequence, timeline

**DIRECTIONAL COVERAGE** — What fraction of 360 degrees the boundary covers. Partial coverage means the shape doesn't extend in all directions — combined with gap direction, this reveals which side is missing. If coverage gap and turning function both point the same way → something is hidden behind something else (partial occlusion).

---

## When Things Contradict

Positive and negative space should tell compatible stories. When they disagree, investigate — or consider that BOTH carry meaning.

Width profile showing gaps on shapes marked "separate" → they may actually tile together as one composite shape.

Grid showing one connected fill but enrichment listing multiple elements → the visual result is one shape built from pieces. Analyze the composite, not the pieces.

Values near clean angle multiples (15, 30, 45, 90) → tool precision, snap to clean value. Values NOT near clean multiples → intentional hand-placement, keep precise.

Nearly identical per-element features across multiple elements → repeated template shape. Check if enrichment already caught this in REPEATED ELEMENTS.

A shape measure saying "simple" but the grid showing complexity → you're looking at the wrong level. Switch from individual elements to the composite, or vice versa.

---

## Traps

**Figure-ground inversion** — Always check what the gaps form, not just the fills.

**Sub-path ≠ separate object** — SVG sub-paths are usually construction pieces of ONE design. Shared bounding region + matching symmetry + recognizable combined outline = parts of one thing.

**Construction ≠ visual** — 3 path elements can produce 1 visible shape. Always check the composite/silhouette.

**Repetition ≠ complexity** — 100 identical dots is simpler than 5 unique organic shapes.

**No single feature determines identity** — Circularity alone cannot distinguish a head from a fruit from a wheel. Synthesize shape + symmetry + spatial context + containment + relative size + arrangement.

**Stroke vs fill** — A stroke-rendered circle is still a circle. Fill percentage reflects technique, not perception.

**Symmetry score alone ≠ orientation** — Moderate symmetry can mean many things.

**Static shapes can imply motion** — A leaning figure suggests falling, a spiral suggests spinning. Don't assume static poses from geometry alone.

**The skeleton reveals structure** — A branching skeleton = articulated parts (limbs, fingers). Single axis = simple elongated form. Radial axes = body with extensions. If medial axis data is available, it's the most compact topology summary.

# Plan: Scene Walkthrough — Co-locate Element Data by Spatial Hierarchy

## Context

The v11 enrichment scatters information about each element across 6+ sections: RECONSTRUCTION lists elements in build order, KEY ELEMENTS lists metrics, SPATIAL CONNECTIONS lists touching pairs, CONTAINMENT TREE shows hierarchy, KEY OVERLAPS shows IoU values. The LLM must mentally grep across 300+ lines to assemble a picture of E40 (acorn shape).

The user's insight: break complex SVGs into sub-problems (like small icons) that LLMs can individually identify, then compose the answer. All properties of each element should be co-located — shape, metrics, color, containment children, touching neighbors, overlaps — in a single walk from outside to inside.

## What Changes

Replace 5 sections (RECONSTRUCTION, KEY ELEMENTS, SPATIAL CONNECTIONS, CONTAINMENT TREE, KEY OVERLAPS) with one **SCENE WALKTHROUGH** section. Remove PIXEL DECOMPOSITION entirely (R-IDs vs E-IDs create competing coordinate systems). Keep everything else unchanged.

### Before (v11 sections):
1. Header
2. SPATIAL INTERPRETATION
3. SHAPE NARRATIVE
4. VISUAL PYRAMID
5. RECONSTRUCTION
6. KEY ELEMENTS
7. SPATIAL CONNECTIONS
8. STRUCTURE (symmetry, color, containment tree, key overlaps, repeated)
9. PIXEL DECOMPOSITION
10. LEARNED PATTERNS

### After (v12 sections):
1. Header
2. SPATIAL INTERPRETATION
3. SHAPE NARRATIVE
4. VISUAL PYRAMID
5. **SCENE WALKTHROUGH** (NEW — replaces 5-8)
6. STRUCTURE (symmetry, color, repeated — no containment tree, no key overlaps)
7. LEARNED PATTERNS

---

## Change 1: New `_build_scene_walkthrough()` Function

**File:** `backend/app/llm/enrichment_formatter.py`

Replace `_build_reconstruction_steps()` (lines 403-558) and `_build_containment_tree()` (lines 566-631) with a single `_build_scene_walkthrough()` function.

### Algorithm

**Complex SVGs** (with primary boundary + containment matrix):

```
SCENE WALKTHROUGH (outside → inside):
  Background: E1 [LARGE] organic at bottom-center, covers 97%×58%, circ=0.16, color=red(#E65270)
    → children: E2, E29(→E30), E40(→E41→E57), E50, E51, ...
    → touches: E7(0.1px), E12(0.1px)
    → overlaps: (none above IoU threshold)

  Primary boundary: E62 [LARGE] organic, covers 100%×100%, circ=0.03, color=black

  Inside E62, feature group [E10+E55+E11] at right (3 concentric layers, max circ=1.00):
    E10 [MEDIUM] elliptical, circ=0.86, color=white(#F9E0E7), salience=33
    E55 [MEDIUM] circular, circ=1.00, salience=40
    E11 [SMALL] organic, salience=?
    → touches: (peer adjacency only)

  Inside E62, feature group [E40+E41+E50+E51] at bottom-right (4 layers, max circ=0.86):
    E40 [LARGE] circular, circ=0.86, salience=40
    E41 [LARGE] organic, color=white(#F6E8A0), salience=34
      → children: E57
    E50 [MEDIUM] organic
    E51 [MEDIUM] organic
    → overlaps: E50↔E51 (IoU=1.000)

  Inside E62, individual: E20 [MEDIUM] organic at top-right, circ=0.33
    → touches: E48(0.5px)

  + 3 small detail elements

  Outside primary: E37 at top-left (spatially isolated)
```

### Data sources (all already computed):
- **Containment hierarchy**: `ctx.containment_matrix` → find direct parent (smallest container) → build children map
- **Concentric groups**: `ctx.subpaths[0].features["concentric_groups"]`
- **Touching pairs**: `ctx.subpaths[0].features["touching_pairs"]` — **filter ancestor-descendant pairs** using containment matrix
- **Overlaps**: `sp.features["overlaps"]` per element
- **Importance scores**: Already computed in `context_to_enrichment_text()` (reuse `importance_scores` dict)
- **Color**: `sp.features["color_label"]`, `sp.features["fill_color"]`
- **Shape**: `sp.features["shape_class"]`, `sp.features["turning_classification"]`
- **Position**: `_region_label()` (already exists at line 214)
- **Entropy caps**: `_entropy_cap()` (line 78), `_coverage_cap()` (line 100)

### Walk order for complex SVGs:
1. **Background elements** — LARGE elements not inside primary boundary (or ≥90% its area)
2. **Primary boundary** — the figure outline (E62)
3. **Concentric groups inside primary** — sorted by member count desc, entropy-capped. Each group on one block with all members listed inline, then → children, → touches, → overlaps
4. **Remaining interior individuals** — LARGE/MEDIUM elements inside primary not in any concentric group, entropy-capped by importance
5. **Small detail summary** — count of remaining small elements
6. **Outside primary** — elements not inside primary and not background, including isolated

### Walk order for simple SVGs (no primary boundary):
- Elements sorted by area, each with full inline data. Same format but no nesting hierarchy.

### Touching pairs filtering:
Build an `ancestors` set for each element from containment matrix. A touching pair (A, B) is only shown if neither A is an ancestor of B nor B is an ancestor of A. This removes E1↔E7 style pairs where E7 is simply nested inside E1.

```python
# Build ancestor sets from containment matrix
ancestors: dict[int, set[int]] = {i: set() for i in range(n)}
for i in range(n):
    for j in range(n):
        if i != j and cmat[i][j]:  # i contains j
            ancestors[j].add(i)
            # Transitive: j inherits i's ancestors
            ancestors[j].update(ancestors[i])
```

### Per-element inline format:
```
  E40 [LARGE] circular(semicircle_or_arc), circ=0.86, conv=0.95, aspect=1.17, color=red, salience=40
```
One line per element. Children shown as `→ children: E57, E30`. Touches shown as `→ touches: E48(0.5px)`. Overlaps shown as `→ overlaps: E50↔E51(IoU=1.000)`.

---

## Change 2: Remove RECONSTRUCTION, KEY ELEMENTS, SPATIAL CONNECTIONS from Main Function

**File:** `backend/app/llm/enrichment_formatter.py`

In `context_to_enrichment_text()` (line 639):
- Remove SECTION 3 (RECONSTRUCTION, lines 721-729) — replaced by walkthrough
- Remove SECTION 4 (KEY ELEMENTS, lines 731-784) — replaced by walkthrough
- Remove SECTION 5 (SPATIAL CONNECTIONS, lines 786-859) — replaced by walkthrough
- Remove CONTAINMENT TREE from SECTION 6 (lines 936-938) — replaced by walkthrough
- Remove KEY OVERLAPS from SECTION 6 (lines 940-957) — replaced by walkthrough
- Add single call: `lines.extend(_build_scene_walkthrough(ctx, sorted_sps, primary_boundary, concentric, is_complex, importance_scores, top_n))`
- Remove SECTION 7 (PIXEL DECOMPOSITION, lines 991-1024) — drop entirely

### Delete unused functions:
- `_build_reconstruction_steps()` (lines 403-558) — fully replaced
- `_build_containment_tree()` (lines 566-631) — fully replaced

---

## Change 3: Filter Touching Pairs (Ancestor-Descendant)

**File:** `backend/app/llm/enrichment_formatter.py` (inside `_build_scene_walkthrough()`)

When displaying touching pairs for an element, exclude any pair where one element is an ancestor/descendant of the other in the containment hierarchy. This fixes the v11 problem where all 10 shown pairs were E1↔child.

---

## Change 4: Update Docstring

**File:** `backend/app/llm/enrichment_formatter.py`

Update the `context_to_enrichment_text()` docstring (lines 640-655) to reflect 6-section layout:
1. Spatial Interpretation
2. Shape Narrative
3. Visual Pyramid
4. Scene Walkthrough
5. Structure (symmetry, color, repeated)
6. Learned Patterns

---

## Files Modified

| File | Changes |
|------|---------|
| `backend/app/llm/enrichment_formatter.py` | Replace `_build_reconstruction_steps` + `_build_containment_tree` with `_build_scene_walkthrough`. Remove 5 section blocks from main function. Remove pixel seg. Update docstring. |

One file only. No changes to transforms, interpreter, or other modules.

---

## Existing Functions to Reuse

- `_entropy_cap(values, safety_max)` → line 78 — cap lists by Shannon perplexity
- `_coverage_cap(children_areas, total_area, safety_max)` → line 100 — cumulative area cap
- `_compute_importance(sp, ctx)` → line 118 — visual salience scoring
- `_region_label(cx, cy, cw, ch)` → line 214 — position label ("top-right", "center")
- `_render_visual_pyramid()` → line 248 — unchanged
- `sanitize_tag()` from `app.svg.anonymizer` → line 14 — SVG tag sanitization

---

## Token Budget

| Section | v11 words | v12 words | Change |
|---------|-----------|-----------|--------|
| Spatial Interpretation | ~500 | ~500 | 0 |
| Shape Narrative | ~150 | ~150 | 0 |
| Visual Pyramid | ~350 | ~350 | 0 |
| Reconstruction | ~200 | 0 | -200 |
| Key Elements | ~250 | 0 | -250 |
| Spatial Connections | ~80 | 0 | -80 |
| Containment Tree | ~100 | 0 | -100 |
| Key Overlaps | ~50 | 0 | -50 |
| **Scene Walkthrough** | 0 | ~500-600 | +500-600 |
| Pixel Decomposition | ~800+ | 0 | -800 |
| Structure (sym+color+rep) | ~150 | ~150 | 0 |
| Learned Patterns | ~30 | ~30 | 0 |
| **Total** | ~2660+ | ~1680-1780 | **-880 to -980** |

Net: ~35% smaller enrichment. All data preserved, just reorganized and pixel seg removed.

---

## Verification

1. `cd backend && uv run pytest --tb=short -q` — all tests pass
2. Run pipeline on Flink SVG, save as `enrichment_flink_v12.txt`:
   - RECONSTRUCTION section absent
   - KEY ELEMENTS section absent
   - SPATIAL CONNECTIONS section absent
   - CONTAINMENT TREE absent from STRUCTURE
   - KEY OVERLAPS absent from STRUCTURE
   - PIXEL DECOMPOSITION absent
   - SCENE WALKTHROUGH present with:
     - Background elements (E1, E2, E5)
     - Primary boundary (E62)
     - Concentric groups with inline member data
     - Touching pairs without ancestor-descendant pairs (no E1↔child)
     - Overlaps inline (E50↔E51)
     - Small detail summary
3. Verify no E-ID appears without context (every E-ID has shape+metrics inline)

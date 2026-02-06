# VectorSight

**Giving AI the ability to see what's inside vector graphics**

---

| | |
|---|---|
| **Team** | Cybots |
| **Competition** | The Strange Data Project |
| **Prize Pool** | €5,000 |
| **Timeline** | Jan 31 – Feb 15, 2025 |

---

## TL;DR — What We Do

### The one-liner

> **We transform SVG code into spatial JSON. Same model, better input, correct answers.**

---

### Quick answers

| Question | Answer |
|---|---|
| **What is VectorSight?** | A geometry engine that transforms SVG code into structured spatial data |
| **Is this fine-tuning?** | No. The model stays the same. |
| **Is this RAG?** | No. We don't search a database. |
| **Is this a tool/MCP?** | No. The LLM doesn't call our code. It receives pre-computed data. |
| **What is it then?** | **Data transformation.** SVG code → spatial JSON. |
| **How does the LLM improve?** | Better input. Same model + better data = better answers. |
| **Does it work on any model?** | Yes. Claude, GPT-4, Llama — all improve with our data. |

---

### The transformation

```
BEFORE (baseline):
┌──────────────────────────────────────────────────────┐
│  SVG code ──────────────────────────────────► LLM    │
│                                                      │
│  <circle cx="8" cy="9" r="1.5"/>                     │
│  <circle cx="16" cy="9" r="1.5"/>                    │
│                                                      │
│  LLM thinks: "I see numbers... 8, 9, 16, 9..."       │
│  LLM outputs: "Maybe 8 units apart?" ← WRONG         │
└──────────────────────────────────────────────────────┘

AFTER (with VectorSight):
┌──────────────────────────────────────────────────────┐
│  SVG code ───► [VectorSight] ───► Spatial JSON ──► LLM│
│                                                      │
│  Spatial JSON:                                       │
│  {                                                   │
│    "left-eye": {"center": [8, 9], "radius": 1.5},   │
│    "right-eye": {"center": [16, 9], "radius": 1.5}, │
│    "gap-edge-to-edge": 5,                            │
│    "relationship": "symmetric"                       │
│  }                                                   │
│                                                      │
│  LLM reads: "Gap is 5 units"                         │
│  LLM outputs: "5 units apart" ← CORRECT              │
└──────────────────────────────────────────────────────┘
```

---

### What we are NOT doing

| Approach | Description | Us? |
|---|---|---|
| Fine-tuning | Train model on new data | ❌ No |
| RAG | Search database, retrieve docs | ❌ No |
| Tool / MCP | LLM calls external function | ❌ No |
| Agent skill | LLM decides what to do | ❌ No |

### What we ARE doing

| Step | What happens |
|---|---|
| 1 | User uploads SVG |
| 2 | Geometry engine computes spatial relationships |
| 3 | Output: structured JSON with distances, containment, symmetry |
| 4 | JSON is added to prompt alongside SVG |
| 5 | LLM reads the JSON and answers correctly |

**The model never changes. We change what it sees.**

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Solution](#2-the-solution)
3. [How It Works](#3-how-it-works)
4. [What You Can Do With It](#4-what-you-can-do-with-it)
5. [How AI Gets Better](#5-how-ai-gets-better)
6. [What Makes This Different](#6-what-makes-this-different)
7. [Demo Day](#7-demo-day)
8. [Timeline](#8-timeline)
9. [Tech Stack](#9-tech-stack)
10. [Scoring Against Criteria](#10-scoring-against-criteria)
11. [Future Work](#11-future-work)
12. [Research Landscape](#12-research-landscape)
13. [Open Questions](#13-open-questions)

---

## 1. The Problem

### What humans see vs what AI sees

**You look at a smiley face icon:**
- Two eyes near the top
- Evenly spaced
- Mouth below them
- All inside a circle

**AI reads the same icon:**
```
M12,8 A2,2 0 1,1 12,4
```
Numbers in a line. No meaning.

### The core issue

The spatial information IS in the code. But AI reads it as text and loses the 2D meaning.

It's like reading sheet music without hearing the song.

### A real example

```
User:   "How far apart are the two letters?" (Anthropic logo)
AI:     "Maybe 5 units? I'm not sure which coordinates to compare."

Actual: 3.42 units (14.2% of icon width)
```

The AI has all the coordinates. It cannot figure out the spatial relationships.

### Why this happens

| Human vision | AI reading SVG |
|---|---|
| 2D grid (retina) | 1D text sequence |
| Left stays left | Left, right, above, below get scrambled |
| "Next to" preserved | Spatial relationships destroyed |

### Why no one has fixed it

| Approach | Limitation |
|---|---|
| VDLM (ICLR 2025) | Fixed vocabulary of 9 shapes. Can't say "inside" or "next to" |
| SGP-Bench | Proves the problem exists. Doesn't solve it |
| LLM4SVG | Generates new SVGs. Doesn't understand existing ones |

**VectorSight is the first system that extracts actual spatial relationships from SVG code.**

---

## 2. The Solution

VectorSight is a processing pipeline.

```
Input:  SVG file
        ↓
        Geometry calculations (maths, not AI)
        ↓
Output: Structured spatial breakdown any AI can understand
```

Think of it as **building eyes for AI**.

### Example: Smiley face

**Input SVG:**
```svg
<circle cx="12" cy="12" r="10"/>           <!-- face -->
<circle cx="8" cy="9" r="1.5"/>            <!-- left eye -->
<circle cx="16" cy="9" r="1.5"/>           <!-- right eye -->
<path d="M7,15 Q12,20 17,15"/>             <!-- smile -->
```

**VectorSight output:**

| Category | Data |
|---|---|
| **Shapes** | Shape 1: circle at (12,12), radius 10 — face outline |
| | Shape 2: circle at (8,9), radius 1.5 — left eye |
| | Shape 3: circle at (16,9), radius 1.5 — right eye |
| | Shape 4: curve from (7,15) to (17,15) — smile |
| **Relationships** | Shapes 2, 3, 4 are INSIDE Shape 1 |
| | Shapes 2 and 3 are SYMMETRIC (4 units from centre each) |
| | Shape 4 is BELOW Shapes 2, 3 (6 units under eyes) |
| | Gap between eyes: 5 units (20.8% of width) |
| **Description** | Smiley face with two symmetric eyes and curved smile |

**Now AI can answer accurately:**

```
User:  "How far apart are the eyes?"
AI:    "5 units — evenly placed, 4 units from centre each side."

User:  "Make it wink."
AI:    "Replace right eye circle with: <path d='M14.5,9 Q16,10.5 17.5,9'/>"
```

Without VectorSight: AI guesses.
With VectorSight: AI knows.

---

## 3. How It Works

### Four processing layers

| Layer | What it does | Analogy |
|---|---|---|
| **Layer 1: Parse** | Multi-representation of each shape (coords, properties, direction, relative position) | Describing a building from every angle |
| **Layer 2: Grid** | Multiple 1D-friendly scans of the 2D layout (row, column, region, Hilbert) | Multiple camera angles of the city |
| **Layer 3: Analysis** | Distances, symmetry, containment, alignment between shapes | Computed relationships between buildings |
| **Layer 4: Reasoning** | Step-by-step spatial reasoning scaffold for the LLM | A tour guide explaining how to read the map |

---

### Layer 1 Deep Dive: Multi-Representation Parsing

The 2D→1D problem starts at parsing. Raw SVG coordinates are 1D soup. We generate multiple representations of each shape so the LLM gets many perspectives.

**For each shape, we compute:**

#### Representation 1: Sequential coordinates

```
path-1: (13.79, 3.93) → (20.22, 20.07) → (23.75, 20.07) → (17.32, 3.93) → close
```

#### Representation 2: Segment types

```
path-1: 4 straight lines, 0 curves, closed polygon
path-2: 11 segments (8 lines, 3 curves), closed polygon with sub-path (cutout)
```

#### Representation 3: Geometric properties

```
path-1: bbox x=[13.79, 23.75], center (18.77, 12.00),
        width 9.96, height 16.14, aspect 0.62 (tall narrow), area 56.88
```

#### Representation 4: Directional description

```
path-1: starts top-center-right, diagonal down-left,
        horizontal right at bottom, vertical up to start.
        Overall: narrow parallelogram leaning right
```

#### Representation 5: Relative to canvas

```
path-1: occupies right third of canvas, full height,
        23.4% of total canvas area
```

**All five computed from the same parsed path. Zero AI. Each gives the LLM a different perspective on the same shape.**

---

### Layer 2 Deep Dive: The 2D→1D Problem

LLMs read text as a 1D sequence. A 2D grid becomes a flat stream of tokens. Spatial relationships get lost. This is the same problem that makes SVG code hard for LLMs — but we can solve it with smarter encoding.

**We generate multiple 1D-friendly representations of the same 2D layout:**

#### Scan Method 1: Row scan-line

Describe each row in words. No 2D parsing needed.

```
Row 0:  empty
Row 2:  filled [4-7], gap, filled [9-11]
Row 3:  filled [3-4], gap [5-7], filled [9-10]
Row 13: filled [0-2], gap [3-7], filled [8-14]
```

#### Scan Method 2: Column scan-line

Same thing but reading columns instead of rows.

```
Col 0:  filled rows [11-13]
Col 3:  filled rows [2-8, 11]
Col 9:  filled rows [2-13]
```

#### Scan Method 3: Region map

Divide canvas into named regions, describe in natural language.

```
Top-left:     triangle apex pointing up
Top-right:    narrow shape pointing up
Middle-left:  triangle widens
Middle-right: narrow line continues
Bottom-left:  wide base with triangular cutout
Bottom-right: narrow base
```

#### Scan Method 4: Hilbert curve

A space-filling curve that maps 2D to 1D while preserving locality.
Points near each other in 2D stay near each other in 1D.

```
Normal row scan:   End of row 1 → start of row 2 = big spatial jump
Hilbert curve:     Continuous path through all cells, no jumps
```

Used in databases for spatial indexing. Nobody has applied this to LLM spatial encoding.

#### Scan Method 5: Text grid (supplementary)

Traditional grid. Imperfect but gives rough visual pattern.

```
. . . . X X X X . X X X . . . .
. . . X X . . X . X . X X . . .
. . . X . . . X . X X . X . . .
```

---

#### Why multiple scans matter

| Single scan | Multiple scans |
|---|---|
| One perspective | Multiple perspectives |
| Might miss patterns | Redundant coverage |
| Can't validate understanding | Can test: does LLM see same thing from all angles? |

**Rotation invariance test:** If LLM identifies "A" from normal scan, rotated scan, AND Hilbert scan — it genuinely understands the shape. Not memorizing.

This is a **novel research contribution**: optimal 2D→1D encoding strategies for LLM spatial reasoning.

---

### Layer 4 Deep Dive: Reasoning Scaffold

We don't just give the LLM data. We guide HOW it thinks about spatial data.

Without a scaffold, the LLM receives a dump of JSON, grids, and numbers. It might use some, ignore some, reason poorly.

With a scaffold, the LLM follows a structured spatial reasoning process.

#### The Spatial Reasoning Protocol

```
When analyzing an SVG with VectorSight data, follow these steps:

STEP 1 — INVENTORY
Read the shape list. How many shapes? What types?
"I see 2 filled paths. path-1 has 4 segments. path-2 has 11 segments."

STEP 2 — LOCATE
Read geometric properties. Where is each shape? How big?
"path-1 is on the right (center 18.77, 12.00), narrow and tall.
 path-2 is on the left (center 8.53, 12.00), wider, with a cutout."

STEP 3 — RELATE
Read relationships. How do shapes connect?
"3.17 units apart. Horizontally aligned. No overlap.
 path-2 is 2.17x the area of path-1."

STEP 4 — VISUALIZE
Read the grid scans. What does the overall layout look like?
"Row scan shows two separate filled regions.
 Region map: triangle with cutout on left, narrow shape on right."

STEP 5 — INTERPRET
Combine all evidence. What is this?
"Two shapes side by side, same height, aligned.
 Left shape: triangle with triangular cutout = likely letter A.
 Right shape: narrow parallelogram = likely letter I.
 Together: 'AI' — possibly a logo."

STEP 6 — ANSWER
Now answer the user's question using all the above.
```

#### Why this matters

| Without scaffold | With scaffold |
|---|---|
| LLM jumps to answer | LLM reasons step by step |
| Might ignore grid data | Uses every representation |
| Inconsistent process | Repeatable, structured |
| "Maybe 5 units?" | "3.17 units (from relationship data)" |

#### This is different from VoT

| VoT | Our scaffold |
|---|---|
| "Imagine a grid as you think" | "Follow these 6 steps using our pre-computed data" |
| LLM creates its own (rough) map | LLM reads our (exact) map |
| One technique | Multi-step structured process |
| Generic spatial reasoning | Specific to SVG + our data format |

**VoT is one line in a prompt. Our scaffold is a full reasoning protocol designed for spatial analysis.**

We can combine both: follow our scaffold AND visualize at each step.

---

### Output columns

| Column | Contents | Method | AI? |
|---|---|---|---|
| **1 — Input** | Raw SVG code | Stored | No |
| **2 — Annotated SVG** | Original code + auto-generated IDs per path | Parser | No |
| **3 — Parse Representations** | Sequential, segment types, geometric, directional, relative | svgpathtools + math | No |
| **4 — Grid Scans** | Row scan, column scan, region map, Hilbert curve, text grid | Computed | No |
| **5 — Relationships** | Distances, containment, symmetry, alignment | Shapely + numpy | No |
| **6 — Style Rules** | Stroke width, fill, CSS separated from structure | XML parser | No |
| **7 — Reasoning Scaffold** | Step-by-step spatial reasoning protocol | Pre-designed prompt template | No |
| **8 — Description** | What it looks like, inferred from columns 2-7 | LLM follows scaffold | Yes |

**Columns 1-7: zero AI.** Computed, deterministic, designed.

**Column 8:** LLM follows our reasoning scaffold with our data. This is the proof that our approach works.

---

## 4. What You Can Do With It

### App 1: Analyzer & Chatbot (main demo)

| Left panel | Right panel | Bottom panel |
|---|---|---|
| SVG with toggleable overlays | Chatbot with spatial context | Tabs for all 8 columns |
| Bounding boxes | "How thick are the lines?" | Raw data, grid, analysis |
| Gap measurements | "Is it balanced?" | Description, clean code |
| Symmetry lines | "Move the eyes closer" | Quality check |

### App 2: Modifier

- Describe changes in plain English
- "Make strokes thicker"
- "Add a hat on top"
- Live before/after preview

### App 3: Icon Set Expander

- Upload icon set
- VectorSight extracts shared rules (2px lines, 12% padding, etc.)
- Generate new icons that match the style

### App 4: Icon Creator

- Describe what you want
- VectorSight generates with spatial awareness
- "Roof is 45% of height — typical is 30-40%, adjust?"

### App 5: SVG Playground

Click any part of any icon. AI knows what it is and responds:

| Click | Response |
|---|---|
| Smiley eye | Winks |
| Smiley mouth | Changes to frown, "O", tongue |
| House door | Opens |
| House chimney | Smoke comes out |
| Car wheel | Spins |

**This is the fun demo.** Judges click, things happen, proves the system understands structure.

---

## 5. How AI Gets Better

The spatial module (Layers 1-3) is just maths. No learning.

AI improvement happens through **data transformation**, not tools or training.

### What we are NOT doing

| Approach | Description | Why not us |
|---|---|---|
| **Fine-tuning** | Change model weights with training data | No training. Model stays the same. |
| **RAG** | Search vector database, retrieve documents | No database. We compute fresh. |
| **Tool/MCP** | LLM calls external function at runtime | No tool calls. LLM receives pre-computed data. |
| **Agent skills** | LLM decides which capability to use | No decision. Data is already transformed. |

### What we ARE doing

**Data representation transformation.**

```
Raw SVG code         →  Geometry engine  →  Structured spatial JSON
(LLM can't use this)     (our contribution)   (LLM can read this)
```

The model doesn't change. The model doesn't call tools.
The model receives a **better representation** of the same data.

### Why this matters for the hackathon

From the competition brief:

> **"Representation & Transformation"** — converting non-text data into model-usable representations

> **"Reinterpreting Existing Data"** — reformatting, restructuring or re-contextualising to extract higher-quality signal

This is exactly what VectorSight does:
- **Input:** SVG code (text, but spatially meaningless to LLMs)
- **Transformation:** Geometry calculations extract spatial relationships
- **Output:** Structured JSON (spatially meaningful, LLMs can read it)

---

### How we give data to the model

### Method 1: Context injection (no model changes)

Paste VectorSight's spatial analysis into the prompt alongside the SVG.

```
Before:
"Here's an SVG. How far apart are the shapes?"
→ AI guesses wrong

After:
"Here's an SVG.
SPATIAL DATA: Shape 1 right edge at x=14.21. Shape 2 left edge at x=10.79. Gap: 3.42 units.
How far apart are the shapes?"
→ AI answers: "3.42 units"
```

No training. Works immediately.

### Method 2: VoT prompting (no model changes)

Microsoft's Visualization-of-Thought (NeurIPS 2024): ask AI to "draw a map" as it reasons.

VoT alone: 27% improvement on spatial tasks.

VoT + VectorSight data: AI gets exact map AND visualizes as it reasons.

```
Prompt addition:
"Visualize each shape on a grid as you reason through this."
```

Free improvement. One extra line in prompt.

### What we deliver

| Priority | Deliverable |
|---|---|
| **Must have** | Spatial JSON + row/column scan-lines + region map |
| **Must have** | Context injection (Method 1) + VoT prompting (Method 2) |
| **Should have** | Hilbert curve scan + rotation invariance testing |
| **Demo shows** | Baseline → VoT alone → VectorSight → VectorSight + VoT |
| **Research shows** | Which scan method gives best LLM accuracy |

---

### Example: "Make it wink"

A user uploads a smiley face and says "make it wink."

**Without VectorSight:**
- AI sees: `<circle cx="16" cy="9" r="1.5"/>`
- AI thinks: "This is... a circle somewhere?"
- AI might delete it, move it wrong, or do nothing

**With VectorSight (Method 1):**
- AI sees spatial context: "circle at (16,9), radius 1.5, INSIDE face outline, SYMMETRIC with circle at (8,9), curve BELOW at y=15"
- Plus description: "right eye"
- AI knows: "This is the right eye. To make it wink, replace the circle with a curved line"
- AI outputs: `<path d='M14.5,9 Q16,10.5 17.5,9'/>`

**With VectorSight + VoT (Method 2):**
- Same as above, but AI also visualizes the layout as it reasons
- More confident, more accurate positioning of the wink curve

---

## 6. What Makes This Different

### Core innovation

Every previous approach either:
- Makes new SVGs without understanding existing ones
- Proves AI can't understand SVGs without offering a solution
- Uses fixed vocabulary that breaks on complex shapes

**VectorSight extracts actual spatial relationships using geometry calculations.**

### Eight unique capabilities

| # | Capability | What it means | Others do this? |
|---|---|---|---|
| 1 | **Multi-representation parsing** | 5 representations per shape (sequential, segment types, geometric, directional, relative) — LLM gets multiple perspectives on the same path | No |
| 2 | **Topology-preserving map** | Tracks what's inside what, next to what, overlapping | No |
| 3 | **Geometry calculations** | Exact distances, thickness, curves, containment | No (others guess or use AI) |
| 4 | **Multi-scan 2D→1D encoding** | Row scan, column scan, region map, Hilbert curve — multiple 1D-friendly representations of 2D layout | No |
| 5 | **Rotation invariance testing** | Validate understanding from multiple scan angles — if LLM recognizes shape from all scans, it truly understands | No |
| 6 | **SVG annotation** | Auto-generate IDs for each path, separate structure from style | No |
| 7 | **Spatial reasoning scaffold** | 6-step structured protocol guiding HOW the LLM reasons about spatial data (Inventory → Locate → Relate → Visualize → Interpret → Answer) | No |
| 8 | **Description from spatial data only** | LLM follows our scaffold with our data to identify "smiley face" from geometry alone, no vision model | No |

**Capabilities 1, 4, and 7 are novel research contributions.**
- **#1 Multi-representation parsing:** Nobody generates multiple complementary descriptions of the same SVG path for LLM consumption.
- **#4 Multi-scan encoding:** Nobody has explored optimal 2D→1D encoding strategies for LLM spatial reasoning. Hilbert curve scanning preserves spatial locality — points near each other in 2D remain near each other in 1D. Established in database spatial indexing, never applied to LLM prompting.
- **#7 Spatial reasoning scaffold:** Goes beyond VoT's single-line prompt. A full 6-step protocol designed specifically for SVG spatial analysis, guiding the LLM through inventory, location, relationships, visualization, interpretation, and answer.

---

## 7. Demo Day

### Pitch (30 seconds)

> "AI reads vector graphics but can't see them. The spatial information is in the code — distances, alignment, containment — but invisible when read as text. VectorSight gives AI eyes."

---

### How Judges Will Test Us

Judges will test our system against **multiple baseline models** to verify improvement.

#### The test setup

```
┌─────────────────────────────────────────────────────────────┐
│  BASELINE TEST (without VectorSight)                        │
│                                                             │
│  Input:   [SVG code] + "How far apart are the eyes?"        │
│  Model:   Claude / GPT-4 / Llama                            │
│  Output:  "Maybe 8 units?" ← WRONG (guessing)               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  WITH VECTORSIGHT (same model, better input)                │
│                                                             │
│  Input:   [SVG code] + [Spatial JSON] + "How far apart..."  │
│  Model:   Same Claude / GPT-4 / Llama                       │
│  Output:  "5 units edge-to-edge" ← CORRECT                  │
└─────────────────────────────────────────────────────────────┘
```

#### What judges are looking for

| Question | How we answer it |
|---|---|
| Does it work on Claude? | Yes — baseline fails, with VectorSight succeeds |
| Does it work on GPT-4? | Yes — same improvement |
| Does it work on Llama? | Yes — same improvement |
| Is it model-specific? | No — works on any model |
| Is it a tool/MCP? | No — model doesn't call anything, just receives better data |

#### Why multiple models matter

| If only works on Claude | If works on all models |
|---|---|
| Maybe just clever prompting | Data representation is genuinely useful |
| Model-specific trick | Model-agnostic improvement |
| Limited value | Broad applicability |

**We want to prove: any model + our spatial data = better performance.**

---

### Demo Script (3 minutes)

**Part 1 — Show the problem (30 sec)**

| Step | What happens |
|---|---|
| 1 | Paste SVG into standard Claude |
| 2 | Ask: "How far apart are the eyes?" |
| 3 | Claude guesses wrong |
| 4 | "This is the baseline. The model can't interpret spatial relationships." |

**Part 2 — Show VectorSight transformation (45 sec)**

| Step | What happens |
|---|---|
| 1 | Upload same SVG to our Analyzer |
| 2 | Show the spatial JSON we generate |
| 3 | Toggle overlays: bounding boxes, gaps, symmetry |
| 4 | "This is what we compute. Pure geometry, no AI." |

**Part 3 — Show improvement across models (90 sec)**

| Model | Baseline | + VectorSight | Improvement |
|---|---|---|---|
| Claude | ❌ Guesses | ✅ Correct | Same model, better input |
| GPT-4 | ❌ Guesses | ✅ Correct | Works here too |
| Llama | ❌ Guesses | ✅ Correct | Model-agnostic |

"The model doesn't change. The data representation does."

**Part 4 — Show scale (15 sec)**

| Point | Message |
|---|---|
| Volume | "5 SVGs → 400+ spatial data points" |
| Availability | "Millions of public SVGs exist" |
| Automation | "Pipeline processes any SVG automatically" |

---

### Close (15 seconds)

> "We didn't just find strange data. We built the eyes to see it."
>
> "Any model. Any SVG. Better answers."

---

## 8. Timeline

### Week 1: Feb 1–7 (Build the pipeline)

| Days | Task |
|---|---|
| 1-2 | SVG parser: read files, convert to absolute coords, separate styling |
| 3-4 | Spatial layers: map (Layer 1), grid (Layer 2), analysis (Layer 3) |
| 5-6 | AI integration: vision descriptions, code rewriter, pixel comparison |
| 7 | Process 5-10 SVGs end-to-end. Fix issues. |

### Week 2: Feb 8–14 (Build the demo)

| Days | Task |
|---|---|
| 8-9 | Build Analyzer app: upload, overlays, chatbot |
| 10-11 | Connect frontend to backend. Polish UI. |
| 12 | Build SVG Playground (click = action) |
| 13 | Run benchmarks: baseline vs VoT vs VectorSight vs combined. Get numbers. |
| 14 | Polish. Record backup video. Submit. |

---

## 9. Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| SVG parsing | svgpathtools | Parse paths into coordinates | No AI |
| Spatial map | Shapely | Containment, intersections, distances | No AI |
| Geometry | numpy, scipy | Symmetry, curvature, measurements | No AI |
| Grid scans | Custom Python | Row scan, column scan, region map, Hilbert curve | No AI |
| Style parsing | xml.etree | Extract stroke/fill/CSS from SVG attributes | No AI |
| Descriptions | Claude API | Interpret spatial data into description | LLM |
| Frontend | Next.js + Tailwind | Web interface | - |
| Backend API | FastAPI | Serve spatial analysis | - |
| Database | Supabase | Store processed SVGs | - |
| Hosting | Vercel + Railway | Frontend + backend | - |

---

## 10. Scoring Against Criteria

| Criteria | Weight | Our approach |
|---|---|---|
| **Demonstrated Improvement** | 25% | Four measurable levels with accuracy numbers. Benchmarked against SGP-Bench and SVGenius. Same model, different inputs, clear performance gap. |
| **Data Novelty** | 20% | Spatial relationships extracted from SVG code. This data EXISTS in every SVG but is inaccessible to LLMs. We make it accessible through geometry computation. |
| **Representation Quality** | 20% | Three layers: spatial map, multi-scan grid (row, column, region, Hilbert curve), relationship analysis. Multiple 1D-friendly encodings of 2D data — novel approach to the 2D→1D problem LLMs face. Deterministic geometry, not AI guessing. |
| **Future Potential** | 20% | Millions of public SVGs. Automated pipeline. Same approach extends to CSS layouts, CAD files, architectural drawings. Foundation for future SVG encoder research. |
| **Clarity of Explanation** | 15% | "AI reads vector graphics but can't see them. VectorSight gives it eyes." Not a tool, not RAG, not fine-tuning. Data transformation that unlocks hidden spatial signal. |

---

## 11. Future Work

### After hackathon

| Timeframe | Goal |
|---|---|
| **Near term** | Process 10,000+ SVGs from public libraries (Heroicons, Material Design, Font Awesome) |
| **Near term** | Full support for curves, gradients, layered shapes |
| **Medium term** | Extend to illustrations, diagrams, charts |
| **Medium term** | Spatial embeddings (continuous representations, not just text) |
| **Medium term** | Apps 2-5 fully built |

### Long term: Fine-tuning

Train open models (Llama, Mistral) on VectorSight's dataset:
- Input: spatial data about shapes
- Output: what it looks like, clean code, correct answers

After training, model learns patterns permanently. Doesn't need full context in every prompt.

**Not required for hackathon. Future research direction.**

### Long term: SVG encoder

Build a dedicated neural network for SVG spatial understanding:
- Vision encoders process images before the language model sees them
- SVG encoder would do the same for vector graphics
- VectorSight's geometry engine = teacher (always exact)
- Encoder = student (learns to approximate fast)

Our dataset is the foundation for this new AI architecture.

### Long term: Expansion

- VectorSight as standard tool for any AI
- Extend to CSS layouts, CAD files, maps, architectural drawings
- Commercial product: "Design tools powered by spatial AI"
- Research publication

---

## 12. Research Landscape

This section explains existing research. For each paper: what they did, why it matters, and what's missing.

---

### 12.1 Papers that tried to solve this problem

#### VDLM (Wang et al., 2024)

**Published:** ICLR 2025

**What they did:**
- Convert images to SVG (using a tool called VTracer)
- Use Mistral-7B to describe shapes
- Output: list of 9 shape types (circle, ellipse, rectangle, triangle, polygon, line, grid, path, graph)
- Trained on 160,000 synthetic image-SVG pairs

**What's good:**
- First attempt at structured SVG understanding
- Shows the problem is solvable

**What's missing:**
- Fixed vocabulary of only 9 shapes
- A semicircle gets called "ellipse" (wrong)
- No relationships between shapes
- Cannot say "this circle is inside that rectangle"
- Cannot say "these two shapes are next to each other"
- Produces flat lists, no topology
- Starts from raster images (lossy), not native SVG
- Synthetic training data = limited complexity
- Their spatial reasoning module actually hurts performance on some tests

**Why this matters for us:**
- VDLM's own "Future Work" section describes what we build
- We use exact geometry where they use approximate AI prediction
- We process real SVG code where they convert from images
- We preserve spatial relationships where they produce flat lists

---

#### LLM4SVG (Xing et al., 2024)

**Published:** CVPR 2025

**What they did:**
- Add 55 special tokens to LLMs for SVG understanding
- Dataset: 250,000 SVGs + 580,000 instruction pairs
- Focus: generate new SVGs from text descriptions

**What's good:**
- Large dataset
- Improves how AI reads SVG tokens

**What's missing:**
- Generation only (make new SVGs)
- Does not understand existing SVGs
- No spatial representation
- No spatial reasoning
- No decompilation (cleaning up messy code)

**Why this matters for us:**
- Complementary work, not competing
- They improve how AI reads SVG tokens
- We improve what spatial information is available

---

### 12.2 Papers that prove the problem exists

#### SGP-Bench (Qiu et al., 2024)

**Published:** ICLR 2025 (Spotlight paper = top 5%)

**What they did:**
- Created benchmark to test if AI can understand SVG programs
- Made simple SVGs that render as handwritten digits (0-9)
- Asked AI: "What digit does this SVG draw?"

**What they found:**
- AI models score 10% accuracy
- 10% = random guessing (there are 10 digits)
- AI literally cannot tell what the SVG draws
- Quote: "SVG programs are quite difficult for LLMs... they fail dramatically"

**Why this matters for us:**
- Validates our core premise
- We use SGP-Bench test cases to show VectorSight's improvement
- Baseline: 10% → With VectorSight: should be much higher

---

#### SVGenius (Chen et al., 2025)

**What they did:**
- Large benchmark: 2,377 queries, 8 task types, 22 AI models tested
- Measures how performance changes as SVGs get more complex

**What they found:**
- All models degrade as complexity increases
- Claude-3.7 Sonnet: 80% accuracy on simple icons → 33% on complex ones
- Quote: "Fundamental limitations in current approaches"

**Why this matters for us:**
- Our primary demo benchmark
- We show: baseline models collapse on complex SVGs
- With VectorSight spatial data: performance stays stable

---

### 12.3 Techniques we use

#### VoT — Visualization-of-Thought (Wu et al., 2024)

**Published:** NeurIPS 2024 (Microsoft Research)

**The idea:**
- Ask AI to "draw a map" as it thinks through spatial problems
- Instead of just thinking in words, AI sketches a grid at each step
- Like asking someone to draw on a napkin while giving directions

**Results:**
- 27% improvement on spatial reasoning tasks
- Beat multimodal models that can actually see images

**How we use it:**
- Add one line to prompts: "Visualize each shape on a grid as you reason"
- Layer on top of VectorSight's exact spatial data
- VoT alone = rough maps (AI draws them itself, sometimes wrong)
- VoT + VectorSight = exact maps + visualization = best results

**Cost:** Free. Just a prompt change. No code, no training.

---

#### Spatial-RAG (Yu et al., 2025)

**The idea:**
- RAG = Retrieval-Augmented Generation
- Connect AI to a spatial database
- AI can answer questions like "find restaurants along this route"

**How it works:**
```
User question → Retrieve spatial data → Feed to AI → Answer
```

**Why this matters for us:**
- Same architecture pattern, different domain
- Spatial-RAG: AI + map database = AI understands geography
- VectorSight: AI + geometry engine = AI understands icons
- Proves our pattern works

---

### 12.4 The SVG encoder gap

**Background:**
- When AI looks at a photo, there's a separate neural network (vision encoder)
- This encoder processes the image BEFORE the language model sees it
- The language model never touches raw pixels
- It gets pre-processed visual information

**The problem:**
- Multiple papers independently note: "SVGs need their own encoder"
- A dedicated neural network that processes SVG code into spatial representations
- One paper states: "in SVG, we would prefer to have a specific embedding module for each geometric primitive"
- LLaVA-SP (ICCV 2025) built spatial-specific tokens for images
- Nobody has built the equivalent for SVG code

**Why this matters for us:**
- VectorSight's geometry engine produces perfect spatial data from SVGs
- Thousands of pairs (SVG code → spatial relationships) = training data for an SVG encoder
- Geometry engine = teacher (always exact)
- Encoder = student (learns to approximate fast)
- Our dataset is the foundation for this new AI architecture

---

### 12.5 Other related work

| Project | What it does | Relevance to us |
|---|---|---|
| **StarVector** | Image-to-SVG generation, 2M dataset | Generation only |
| **OmniSVG** | Vision-language model for SVG generation, 2M dataset | Generation only |
| **Chat2SVG** | Text-to-SVG via templates and optimization | Generation only |
| **UniSVG** | Multi-task SVG framework, 360K SVGs | Generation + understanding, but no structured spatial representation |
| **SVGEditBench** | Benchmark for SVG editing accuracy | Benchmark only |
| **VGBench** | Benchmark across SVG, TikZ, and Graphviz | Benchmark only |
| **SpatialLLM** (CVPR 2025) | 3D spatial understanding for multimodal LLMs | Same principle as ours, but for 3D scenes from images |

---

### 12.6 Summary: The gap we fill

| What others do | What's missing |
|---|---|
| Generate new SVGs | Don't understand existing ones |
| Prove AI fails on SVGs | Don't offer a solution |
| Describe shapes with fixed vocabulary | No relationships, no topology |
| Use AI prediction | Can be wrong, not deterministic |

**VectorSight is the first system that:**
- Extracts topology-preserving spatial relationships
- From actual SVG code (not converted from images)
- Using computational geometry (exact, deterministic)
- Making it available to any AI model

---

## 13. Open Questions

### Grid & Scan Research

| Question | Options to explore |
|---|---|
| Grid resolution | 8×8 vs 16×16 vs 24×24 vs adaptive |
| Best scan method | Row scan vs column scan vs Hilbert curve vs region map |
| Scan combinations | Does giving multiple scans improve accuracy? How many is optimal? |
| Rotation invariance | Does LLM identify same shape from rotated/flipped scans? |
| Hilbert vs Z-order | Which space-filling curve preserves locality better for LLMs? |

### General

| Question | Options to explore |
|---|---|
| Geometry calculations | Which give most value per compute cost? |
| Advanced SVG features | How to handle gradients, filters, masks? |
| Complex paths | How to handle bezier curves with many control points? |
| VoT + our scans | Does VoT add value when we already provide pre-computed scans? |

---

**VectorSight by Cybots**

*"AI reads vector graphics but can't see them. We built the eyes."*

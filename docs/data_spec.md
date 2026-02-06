# VectorSight Data Specification

**What data we use, where it comes from, and what we do with it.**

---

## Core Concept: Data Transformation

VectorSight is **not** RAG, fine-tuning, or a tool/MCP.

VectorSight is **data transformation**:

```
Input:   SVG code (coordinates as text — LLMs cannot interpret spatially)
Process: Geometry engine (Shapely, svgpathtools, numpy)
Output:  Spatial JSON (structured relationships — LLMs can read and use)
```

| What | Before transformation | After transformation |
|---|---|---|
| Format | Raw SVG markup | Structured JSON |
| "Eye at (8,9)" | Hidden in path coordinates | Explicit: `{"label": "eye", "center": [8,9]}` |
| "Inside face" | Not stated anywhere | Explicit: `{"type": "contains", "parent": "face"}` |
| "5 units apart" | Must calculate from coords | Explicit: `{"eye-gap": 5}` |

The LLM receives transformed data. No tool calls. No retrieval. No training.

---

## Table of Contents

1. [Icon Sets](#1-icon-sets)
2. [Sample SVGs](#2-sample-svgs)
3. [Data Pipeline](#3-data-pipeline)
4. [Output Schema](#4-output-schema)
5. [Benchmarks](#5-benchmarks)

---

## 1. Icon Sets

### Primary: Lucide Icons

| Property | Value |
|---|---|
| **Name** | Lucide |
| **URL** | https://lucide.dev |
| **GitHub** | https://github.com/lucide-icons/lucide |
| **Count** | 1,400+ icons |
| **Size** | 24x24 viewBox |
| **Style** | Minimal outline, 2px stroke |
| **License** | ISC (permissive, free to use) |
| **Used by** | shadcn/ui, Next.js projects |

**Why Lucide:**
- Clean, consistent SVG output
- No transforms or complex groupings
- Standard 24x24 canvas
- Actively maintained
- Most popular in modern React/Next.js

### Secondary: For variety testing

| Icon Set | URL | Why include |
|---|---|---|
| **Heroicons** | https://heroicons.com | Tailwind ecosystem, outline + solid variants |
| **Feather** | https://feathericons.com | Original minimal set, 280 icons |
| **Radix Icons** | https://icons.radix-ui.com | Smaller 15x15, tests different scale |
| **Simple Icons** | https://simpleicons.org | Brand logos, more complex paths |

---

## 2. Sample SVGs

### 2.1 Test Set (5 icons for development)

These 5 icons cover different complexity levels. Use during development.

#### Icon 1: Circle (simplest)

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10"/>
</svg>
```

| Property | Value |
|---|---|
| **Shapes** | 1 circle |
| **Complexity** | Minimal |
| **Tests** | Basic circle detection |

---

#### Icon 2: Smiley (our running example)

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10"/>
  <circle cx="8" cy="9" r="1"/>
  <circle cx="16" cy="9" r="1"/>
  <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
</svg>
```

| Property | Value |
|---|---|
| **Shapes** | 3 circles + 1 path |
| **Complexity** | Medium |
| **Tests** | Containment, symmetry, curves |

---

#### Icon 3: Home (Lucide)

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/>
  <path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
</svg>
```

| Property | Value |
|---|---|
| **Shapes** | 2 paths (door + house outline) |
| **Complexity** | Medium |
| **Tests** | Path parsing, rectangles from paths, hierarchy |

---

#### Icon 4: Settings/Gear (Lucide)

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
  <circle cx="12" cy="12" r="3"/>
</svg>
```

| Property | Value |
|---|---|
| **Shapes** | 1 complex path + 1 circle |
| **Complexity** | High |
| **Tests** | Complex path parsing, center detection, gear teeth |

---

#### Icon 5: Chart/Analytics (Lucide bar-chart-2)

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <line x1="18" x2="18" y1="20" y2="10"/>
  <line x1="12" x2="12" y1="20" y2="4"/>
  <line x1="6" x2="6" y1="20" y2="14"/>
</svg>
```

| Property | Value |
|---|---|
| **Shapes** | 3 lines |
| **Complexity** | Simple geometry, semantic meaning |
| **Tests** | Line detection, alignment, relative heights |

---

### 2.2 Validation Set (20 icons)

After development, test against these 20 icons from Lucide:

| Category | Icons |
|---|---|
| **Simple shapes** | circle, square, triangle, star |
| **UI elements** | menu, x, check, plus, minus |
| **Objects** | home, settings, user, mail, phone |
| **Arrows** | arrow-right, arrow-up, chevron-down |
| **Complex** | calendar, clock, search, heart |

---

### 2.3 Benchmark Set (100 icons)

For final accuracy numbers, use 100 icons spanning all complexity levels.

| Complexity | Count | Examples |
|---|---|---|
| Simple (1-2 shapes) | 30 | circle, square, line, dot |
| Medium (3-5 shapes) | 40 | home, user, mail, check-circle |
| Complex (6+ shapes) | 30 | settings, calendar, layout-dashboard |

---

## 3. Data Pipeline

### 3.1 Input → Processing → Output

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  SVG File   │ ──► │  VectorSight    │ ──► │  8 Columns  │
│  (raw code) │     │  Geometry Engine│     │  (JSON/DB)  │
└─────────────┘     └─────────────────┘     └─────────────┘
```

### 3.2 Processing Steps

| Step | Input | Output | Tool |
|---|---|---|---|
| 1. Parse | SVG file | Shape objects | svgpathtools |
| 2. Normalize | Shape objects | Absolute coordinates | Python |
| 3. Map | Coordinates | Spatial map (Layer 1) | Custom |
| 4. Grid | Spatial map | Grid overlay (Layer 2) | Custom |
| 5. Analyze | Map + Grid | Relationships (Layer 3) | Shapely |
| 6. Describe | Rendered image | Text description | Claude Vision |
| 7. Clean | Original SVG + analysis | Clean code | Claude |
| 8. Verify | Original + Clean | Match score | PIL |

---

## 4. Output Schema

### 4.1 Per-SVG Output (JSON)

```json
{
  "id": "lucide-home-001",
  "source": {
    "set": "lucide",
    "name": "home",
    "file": "home.svg",
    "original_code": "<svg>...</svg>"
  },
  "shapes": [
    {
      "id": "shape-1",
      "type": "path",
      "label": "house-outline",
      "bounds": { "x": 3, "y": 3, "width": 18, "height": 18 },
      "center": { "x": 12, "y": 12 },
      "area": 180.5
    },
    {
      "id": "shape-2",
      "type": "path",
      "label": "door",
      "bounds": { "x": 9, "y": 13, "width": 6, "height": 8 },
      "center": { "x": 12, "y": 17 },
      "area": 42.0
    }
  ],
  "relationships": [
    {
      "type": "contains",
      "parent": "shape-1",
      "child": "shape-2"
    },
    {
      "type": "alignment",
      "shapes": ["shape-1", "shape-2"],
      "axis": "vertical-center"
    }
  ],
  "grid": {
    "resolution": 16,
    "cells": [
      { "row": 0, "col": 7, "state": "filled" },
      { "row": 0, "col": 8, "state": "filled" }
    ]
  },
  "properties": {
    "symmetry": "vertical",
    "balance": "centered",
    "stroke_width": 2,
    "complexity_score": 3.5
  },
  "description": "House icon with triangular roof and rectangular door centered at bottom",
  "clean_code": "<svg>...</svg>",
  "quality_score": 0.98
}
```

### 4.2 Column Mapping

| Column | JSON Path | Type |
|---|---|---|
| A - Original SVG | `source.original_code` | string |
| B - File Info | `source.*` | object |
| C - Spatial Map | `shapes` + `grid` | array + object |
| D - Shape Analysis | `relationships` + `properties` | array + object |
| E - Description | `description` | string |
| F - Clean Code | `clean_code` | string |
| G - Quality Check | `quality_score` | number (0-1) |
| H - Design Rules | (computed across set) | object |

---

## 5. Benchmarks & Testing

### 5.1 How Judges Will Test

Judges test against **multiple baseline models** to verify our improvement is real and model-agnostic.

#### Test structure

```
For each model (Claude, GPT-4, Llama):
│
├── Test 1: BASELINE
│   Input:  SVG code + question
│   Output: Model's answer
│   Record: Correct or wrong?
│
└── Test 2: WITH VECTORSIGHT
    Input:  SVG code + Spatial JSON + question
    Output: Model's answer
    Record: Correct or wrong?
```

#### Expected results

| Model | Baseline | + VectorSight | What this proves |
|---|---|---|---|
| Claude | ❌ ~10-30% | ✅ >80% | Works on Claude |
| GPT-4 | ❌ ~10-30% | ✅ >80% | Works on GPT-4 |
| Llama | ❌ ~10-30% | ✅ >80% | Works on open models |
| **All** | **Fail** | **Pass** | **Model-agnostic** |

#### Why this matters

| If only works on one model | If works on all models |
|---|---|
| Might be prompt engineering | Data representation is the cause |
| Limited applicability | Broad value |
| Weaker submission | Stronger submission |

---

### 5.2 SGP-Bench Test Cases

From the SGP-Bench paper (ICLR 2025). Simple SVGs that render as digits 0-9.

| Property | Value |
|---|---|
| **Format** | SVG code → "What digit is this?" |
| **Correct answers** | 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9 |
| **Baseline accuracy** | ~10% (random guess level) |
| **Our target** | >80% |

---

### 5.3 SVGenius Test Cases

From the SVGenius benchmark. 8 task types:

| Task | Example Question | Baseline | Target |
|---|---|---|---|
| Shape identification | "What shape is at (12, 12)?" | Poor | High |
| Counting | "How many circles?" | Medium | High |
| Spatial relations | "Is A inside B?" | Poor | High |
| Distance | "How far apart?" | Poor | High |
| Symmetry | "Is this symmetric?" | Poor | High |
| Alignment | "Are these aligned?" | Poor | High |
| Size comparison | "Which is larger?" | Medium | High |
| Description | "Describe this icon" | Medium | High |

**Baseline pattern:** Accuracy drops 80% → 33% as complexity increases.

**Our target:** Stable >70% across all complexity levels.

---

### 5.4 Our Test Protocol

#### The 4 levels

| Level | What model receives | Expected accuracy |
|---|---|---|
| 1 | SVG only | Low (baseline) |
| 2 | SVG + VoT prompt | Slightly better |
| 3 | SVG + Spatial JSON | High |
| 4 | SVG + Spatial JSON + VoT | Highest |

#### Metrics we report

| Metric | What it shows |
|---|---|
| Accuracy per task type | Which spatial skills improve |
| Accuracy vs complexity | Performance stability |
| Cross-model consistency | Same improvement on all models |
| Response consistency | Same question = same answer |

---

### 5.5 Test Assets We Prepare

| Asset | Count | Purpose |
|---|---|---|
| Test SVGs | 10 | Core test cases |
| Spatial JSONs | 10 | Pre-computed for each SVG |
| Questions per SVG | 5-10 | Cover all task types |
| Total test cases | 50-100 | Statistical significance |

---

## 6. Data Storage

### 6.1 Folder Structure

```
samples/
├── test/           # 5 icons for development
│   ├── circle.svg
│   ├── smiley.svg
│   ├── home.svg
│   ├── settings.svg
│   └── bar-chart.svg
├── validation/     # 20 icons for testing
└── benchmark/      # 100 icons for final numbers

processed/
├── test/           # Processed JSON outputs
├── validation/
└── benchmark/
```

### 6.2 Supabase Tables

| Table | Purpose |
|---|---|
| `svgs` | Original SVG files |
| `analyses` | VectorSight output (Columns C-D) |
| `descriptions` | AI-generated descriptions (Column E) |
| `clean_code` | Cleaned SVG output (Column F) |
| `benchmarks` | Test results and accuracy scores |

---

**Next:** See `user_journeys.md` for frontend flows and input types.

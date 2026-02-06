# VectorSight

**Giving AI the ability to see what's inside vector graphics**

---

## The One-Liner

> **We transform SVG code into spatial JSON. Same model, better input, correct answers.**

---

## Team Cybots

| | |
|---|---|
| **Competition** | The Strange Data Project |
| **Prize Pool** | €5,000 |
| **Timeline** | Jan 31 – Feb 15, 2025 |

---

## What We Do

### The Problem

```
LLM sees:     <circle cx="8" cy="9"/><circle cx="16" cy="9"/>
LLM thinks:   "Numbers... 8, 9, 16, 9..."
LLM outputs:  "Maybe 8 units apart?" ← WRONG
```

### Our Solution

```
SVG ──► [VectorSight Geometry Engine] ──► Spatial JSON ──► LLM

Spatial JSON: {"left-eye": [8,9], "right-eye": [16,9], "gap": 5}

LLM reads:    "Gap is 5 units"
LLM outputs:  "5 units apart" ← CORRECT
```

### The Key Point

| What changes | What stays the same |
|---|---|
| Input data (we transform it) | The model (Claude, GPT-4, Llama) |

**Same model. Better data. Better answers.**

---

## What This Is NOT

| Approach | Us? | Why not |
|---|---|---|
| Fine-tuning | ❌ | We don't train or modify any model |
| RAG | ❌ | We don't search a database |
| Tool / MCP | ❌ | LLM doesn't call our code at runtime |
| Agent skill | ❌ | No decision-making by the LLM |

**We are: Data transformation.**

We convert SVG coordinates into structured spatial relationships that any LLM can read.

---

## How Judges Test Us

| Test | Input | Expected Result |
|---|---|---|
| Baseline Claude | SVG only | ❌ Guesses wrong |
| Baseline GPT-4 | SVG only | ❌ Guesses wrong |
| Baseline Llama | SVG only | ❌ Guesses wrong |
| Claude + VectorSight | SVG + Spatial JSON | ✅ Correct |
| GPT-4 + VectorSight | SVG + Spatial JSON | ✅ Correct |
| Llama + VectorSight | SVG + Spatial JSON | ✅ Correct |

**Proves:** Our spatial data helps any model, not just one.

---

## Project Structure

```
vectorsight/
├── docs/
│   ├── prd.md              # Full product requirements
│   ├── data_spec.md        # Data inputs, schemas, benchmarks
│   └── user_journeys.md    # Frontend flows, input types
├── backend/                # Python spatial processing (TODO)
├── frontend/               # Next.js web app (TODO)
├── samples/                # Test SVG files (TODO)
└── README.md
```

---

## Tech Stack

| Component | Tool |
|---|---|
| SVG parsing | Python, svgpathtools |
| Spatial analysis | Shapely, numpy |
| AI integration | Claude API |
| Frontend | Next.js, Tailwind |
| Backend API | FastAPI |
| Database | Supabase |
| Hosting | Vercel + Railway |

---

## Status

| Phase | Status |
|---|---|
| PRD | ✅ Complete |
| Data spec | ✅ Complete |
| User journeys | ✅ Complete |
| Backend: SVG parser | ⬜ TODO |
| Backend: Spatial engine | ⬜ TODO |
| Backend: API | ⬜ TODO |
| Frontend: Analyzer | ⬜ TODO |
| Frontend: Playground | ⬜ TODO |
| Benchmarks | ⬜ TODO |

---

## Quick Links

| Document | What's in it |
|---|---|
| [prd.md](docs/prd.md) | Problem, solution, demo script, research |
| [data_spec.md](docs/data_spec.md) | Icon sets, sample SVGs, output schemas, benchmarks |
| [user_journeys.md](docs/user_journeys.md) | App layouts, user flows, input types |

---

*VectorSight by Cybots — "We built the eyes."*

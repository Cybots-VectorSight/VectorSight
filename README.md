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
| **Prize Pool** | $5,000 |
| **Timeline** | Jan 31 - Feb 15, 2025 |

---

## What We Do

### The Problem

```
LLM sees:     <circle cx="8" cy="9"/><circle cx="16" cy="9"/>
LLM thinks:   "Numbers... 8, 9, 16, 9..."
LLM outputs:  "Maybe 8 units apart?" <- WRONG
```

### Our Solution

```
SVG --> [VectorSight Geometry Engine] --> Spatial JSON --> LLM

Spatial JSON: {"left-eye": [8,9], "right-eye": [16,9], "gap": 5}

LLM reads:    "Gap is 5 units"
LLM outputs:  "5 units apart" <- CORRECT
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
| Fine-tuning | No | We don't train or modify any model |
| RAG | No | We don't search a database |
| Tool / MCP | No | LLM doesn't call our code at runtime |
| Agent skill | No | No decision-making by the LLM |

**We are: Data transformation.**

We convert SVG coordinates into structured spatial relationships that any LLM can read.

---

## Architecture

```
                    Raw SVG
                      |
              [Layer 0: Parse]         7 transforms
                      |
            [Layer 1: Shape Analysis]  21 transforms
                      |
           [Layer 2: Visualization]    7 transforms
                      |
           [Layer 3: Relationships]    21 transforms
                      |
            [Layer 4: Validation]      5 transforms
                      |
               Enrichment Text         ~1,200 tokens
                      |
                 LLM reads it
```

**61 transforms** total, plugin-based with `@transform` decorator and topological dependency sort.

---

## Project Structure

```
vectorsight/
├── docs/
│   ├── prd.md                 # Product requirements
│   ├── data_spec.md           # Data schemas, benchmarks
│   ├── user_journeys.md       # Frontend flows
│   └── vectorsight_guide.md   # Technical guide (61 transforms)
│
├── backend/                   # Python geometry engine
│   ├── app/
│   │   ├── main.py            # FastAPI app
│   │   ├── config.py          # Pydantic Settings
│   │   ├── api/               # 7 API endpoints
│   │   ├── engine/            # Core transform engine
│   │   │   ├── registry.py    # @transform decorator
│   │   │   ├── context.py     # PipelineContext + SubPathData
│   │   │   ├── pipeline.py    # Orchestrator with adaptive gating
│   │   │   ├── layer0/        # SVG parsing (7 transforms)
│   │   │   ├── layer1/        # Shape analysis (21 transforms)
│   │   │   ├── layer2/        # Visualization (7 transforms)
│   │   │   ├── layer3/        # Relationships (21 transforms)
│   │   │   ├── layer4/        # Validation (5 transforms)
│   │   │   └── resolver/      # Intent -> SVG (create/modify)
│   │   ├── llm/               # LangChain + ChatAnthropic
│   │   ├── svg/               # Parser, serializer, anonymizer
│   │   └── utils/             # Geometry, rasterizer, morphology
│   ├── tests/                 # 55 tests across 6 modules
│   ├── pyproject.toml         # uv project
│   └── Dockerfile
│
├── frontend/                  # Next.js + Bun + Hono (single Vercel project)
│   ├── src/app/               # App Router (pages + UI)
│   │   └── api/[[...route]]   # Hono catch-all → Vercel serverless functions
│   ├── vercel.json            # Vercel config (Bun runtime)
│   └── package.json           # Bun + React 19 + Hono + Zod
│
└── .github/workflows/
    └── ci.yml                 # Backend tests + frontend build
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Geometry engine | Python 3.13, svgpathtools, shapely, numpy, scipy, scikit-learn |
| Backend API | FastAPI, uvicorn, Pydantic |
| LLM integration | LangChain, langchain-anthropic (ChatAnthropic) |
| Frontend + API | Next.js 16, React 19, Bun, Hono, Zod, TanStack Query |
| Styling | Tailwind CSS 4 |
| Package manager | uv (backend), Bun (frontend) |
| Deployment | Vercel (Bun runtime, Hono as serverless functions) |
| CI/CD | GitHub Actions |

---

## Getting Started

### Backend

```bash
cd backend

# Install uv (if not installed)
# https://docs.astral.sh/uv/getting-started/installation/

# Install dependencies
uv sync

# Copy env and add your API key
cp .env.example .env

# Run tests
uv run pytest

# Start dev server
uv run uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install Bun (if not installed)
# https://bun.sh/docs/installation

# Install dependencies
bun install

# Start dev server
bun dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/analyze` | Analyze SVG, return enrichment |
| POST | `/api/chat` | Chat about an SVG with spatial context |
| POST | `/api/modify` | Modify existing SVG via natural language |
| POST | `/api/create` | Create new SVG from description |
| POST | `/api/icon-set/analyze` | Batch analyze icon set |
| POST | `/api/playground/enrich` | Raw enrichment for experimentation |

---

## Engine: 61 Transforms

| Layer | Count | Purpose |
|---|---|---|
| L0: Parsing | 7 | SVG normalization, bezier sampling, subpath extraction |
| L1: Shape Analysis | 21 | Curvature, Fourier descriptors, symmetry, corners, shape classification |
| L2: Visualization | 7 | ASCII grid, silhouette, negative space, figure-ground |
| L3: Relationships | 21 | Containment, distance, DBSCAN grouping, stacking tree, patterns |
| L4: Validation | 5 | Canonical orientation, rotation invariance, consistency checks |

Adaptive gating skips irrelevant transforms based on SVG complexity (stroke-only, simple/complex thresholds).

---

## Deployment

Frontend, Hono API, and Bun all deploy as **one Vercel project**.

```
frontend/
├── src/app/                    # Next.js pages (SSR/static)
│   └── api/[[...route]]/       # Hono catch-all route
│       └── route.ts            # handle() from hono/vercel
├── vercel.json                 # Bun install + build
└── package.json
```

- **Hono** runs inside Next.js via `hono/vercel` adapter — each exported HTTP method (`GET`, `POST`, etc.) becomes a Vercel serverless function
- **Bun** is the runtime — Vercel uses it for install and build via `vercel.json`
- **No separate API server** — Hono routes live alongside Next.js pages in the same deploy
- **CI/CD** — GitHub Actions runs `uv run pytest` (backend) and `bun run lint && bun run build` (frontend) on every push/PR

---

## Status

| Phase | Status |
|---|---|
| Documentation | Done |
| Backend: 61 transforms | Done |
| Backend: API endpoints | Done |
| Backend: LLM integration | Done |
| Backend: Intent resolver | Done |
| Backend: 55 tests passing | Done |
| Frontend: Scaffold | Done |
| Frontend: Features | TODO |
| CI/CD | Done |
| Deployment | TODO |

---

## Quick Links

| Document | What's in it |
|---|---|
| [prd.md](docs/prd.md) | Problem, solution, demo script, research |
| [data_spec.md](docs/data_spec.md) | Icon sets, sample SVGs, output schemas, benchmarks |
| [user_journeys.md](docs/user_journeys.md) | App layouts, user flows, input types |
| [vectorsight_guide.md](docs/vectorsight_guide.md) | Technical guide: all 61 transforms, enrichment format |

---

*VectorSight by Cybots -- "We built the eyes."*

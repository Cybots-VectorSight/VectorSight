# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VectorSight — SVG spatial analysis engine that transforms SVG code into structured spatial enrichment text for LLM comprehension. 3-stage breakdown pipeline (Path Separation → Silhouette → Prompt Building) with 14 steps across 5 layers.

## Commands

### Backend (working directory: `backend/`)

```bash
uv sync                                      # Install/update dependencies
uv run pytest                                # Run all tests (68 tests, 4 modules)
uv run pytest tests/test_breakdown/ -q       # Run breakdown tests
uv run pytest tests/test_breakdown/test_pipeline.py::test_name -q  # Run single test
uv run pytest --tb=short -q                  # CI mode
uv run uvicorn app.main:app --reload --port 8003  # Dev server
```

### Frontend (working directory: `frontend/`)

```bash
bun install                  # Install dependencies
bun dev                      # Dev server at localhost:3000
bun run build                # Production build
bun run lint                 # ESLint
```

## Architecture

### Backend: Breakdown Pipeline

SVG → Path Separation (3 steps) → Shape Analysis (4 steps) → Visualization (3 steps) → Relationships (3 steps) → Validation (1 step) → Enrichment Text → LLM

**3 stages:**
1. **Separate** (`app/engine/breakdown/separate.py`) — Split compound SVG paths, convert to polygons, merge overlapping layers via Ochiai coefficient, group by containment tree
2. **Silhouette** (`app/engine/breakdown/silhouette.py`) — B-spline smoothing + Schneider Bezier fitting + spike detection for each group
3. **Prompt Builder** (`app/engine/breakdown/prompt_builder.py`) — Shape descriptors, ASCII/Braille grids, protrusion detection, symmetric pairs, VoT reasoning framework, enrichment text generation

**Core engine classes:**
- `app/engine/breakdown/__init__.py` — `BreakdownResult` dataclass, `run_breakdown()` orchestrator
- `app/engine/breakdown/separate.py` — `GroupData` dataclass, `load_and_split()`, `merge_overlapping()`, `group_by_proximity()`
- `app/engine/breakdown/silhouette.py` — `SilhouetteResult` dataclass, `research_silhouette()`
- `app/engine/breakdown/prompt_builder.py` — `build_enrichment_text()`, `build_enrichment_output()`
- `app/engine/pipeline.py` — `BreakdownPipeline` with `run()` and `run_streaming()`, `create_pipeline()` factory

**14 pipeline steps** (emitted as SSE progress events for frontend):
- B0.01-B0.03: PARSING (load paths, split compounds, convert to polygons)
- B1.01-B1.04: SHAPE_ANALYSIS (similarity matrix, merge overlapping, group by containment, shape descriptors)
- B2.01-B2.03: VISUALIZATION (silhouettes, ASCII grids, Braille outlines)
- B3.01-B3.03: RELATIONSHIPS (protrusions, symmetric pairs, feature roles)
- B4.01: VALIDATION (build enrichment text)

### Backend: Other Key Modules

- `app/svg/parser.py` — Regex-based SVG parsing with order-independent attribute extraction (handles line/circle/rect/path/polyline/polygon/ellipse)
- `app/llm/model_router.py` — Routes to Haiku (cheap), Sonnet (mid/frontier) via `app/config.py` settings
- `app/learning/` — JSONL session memory, self-reflection, knowledge tags, seed patterns
- `app/engine/resolver/` — Intent parsing + SVG generation for create/modify endpoints
- `app/engine/pixel_segmentation.py` — Standalone pixel-based segmentation (not yet integrated)
- `app/svg/anonymizer.py` — `sanitize_for_llm()` strips identifying attributes before LLM sees SVG

### Frontend

- Next.js 16 App Router with route groups: `(public)/` for marketing pages, `(app)/` for workspace
- API proxy: `src/app/api/[[...route]]/route.ts` — Hono catch-all that proxies to the Python backend (`BACKEND_URL` env var, defaults to `http://localhost:8000`)
- Chat UI components in `src/components/prompt-kit/` (streaming responses, file upload, markdown rendering)
- shadcn/ui (new-york style) with Base UI components, Tailwind CSS 4
- State: TanStack Query for server state

### Deployment

- Frontend deploys as one Vercel project (Bun runtime via `vercel.json`). Hono routes become Vercel serverless functions alongside Next.js pages.
- CI: GitHub Actions (`.github/workflows/ci.yml`) — backend: uv + pytest, frontend: bun lint + build

## Important Patterns

- **Pipeline takes SVG string directly:** `create_pipeline().run(svg_text)` — no separate parse step needed, the breakdown pipeline handles SVG parsing internally via `svgpathtools`
- **SVG parsing:** Attribute regexes are order-independent — `cx`, `cy`, `r` can appear in any order on circle/line tags
- **Build system:** Backend uses `uv_build` (not hatchling) with `module-name="app"` and `module-root=""` in pyproject.toml. Dev deps use `[dependency-groups]` not `[project.optional-dependencies]`
- **Test fixtures:** `tests/conftest.py` provides SVG string fixtures: `circle_svg`, `smiley_svg`, `home_svg`, `settings_svg`, `filled_rect_svg`, `filled_complex_svg`
- **Environment:** Backend needs `ANTHROPIC_API_KEY` in `backend/.env`. Frontend needs `BACKEND_URL` in `frontend/.env.local`
- **Generalized vocabulary:** No animal-specific terms in enrichment text — uses generic spatial descriptions (e.g., "primary upper region" instead of "head/face region")

## Documentation

- `docs/vectorsight_guide.md` — Comprehensive technical guide (enrichment format reference)
- `docs/prd.md` — Product requirements, demo script, research landscape
- `docs/data_spec.md` — Input/output schemas, icon sets, benchmarks
- `docs/user_journeys.md` — 5 user flows
- `docs/frontend.md` — Frontend architecture and component guide

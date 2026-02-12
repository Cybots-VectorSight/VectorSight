# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VectorSight — SVG spatial analysis engine that transforms SVG code into structured spatial enrichment text for LLM comprehension. 61 geometry transforms across 5 layers, plugin-based with `@transform` decorator and topological dependency sort.

## Commands

### Backend (working directory: `backend/`)

```bash
uv sync                                      # Install/update dependencies
uv run pytest                                # Run all tests (55 tests, 6 modules)
uv run pytest tests/test_layer1/ -q          # Run one test module
uv run pytest tests/test_engine/test_pipeline.py::test_name -q  # Run single test
uv run pytest --tb=short -q                  # CI mode
uv run uvicorn app.main:app --reload --port 8000  # Dev server
```

### Frontend (working directory: `frontend/`)

```bash
bun install                  # Install dependencies
bun dev                      # Dev server at localhost:3000
bun run build                # Production build
bun run lint                 # ESLint
```

## Architecture

### Backend: Transform Pipeline

SVG → L0 Parse (7) → L1 Shape Analysis (21) → L2 Visualization (7) → L3 Relationships (21) → L4 Validation (5) → Enrichment Text → LLM

**Core engine classes:**
- `app/engine/registry.py` — `@transform` decorator, `TransformRegistry` singleton, `Layer` enum, Kahn's topological sort for dependency resolution
- `app/engine/context.py` — `PipelineContext` (shared state: SVG doc, subpaths, enrichment, config) + `SubPathData` (per-element features dict)
- `app/engine/pipeline.py` — `Pipeline` orchestrator with `AdaptiveGate` (skips irrelevant transforms based on SVG complexity)
- `app/engine/interpreter.py` — Post-pipeline spatial interpretation (silhouette, orientation, mass, protrusions, focal elements)

**Adding a new transform:** Create one file in the appropriate `layer{N}/` directory with the `@transform` decorator. Nothing else needs to change — the registry auto-discovers via `_register_transforms()` in `app/main.py`.

**Transform layers** live in `app/engine/layer{0-4}/`, each file is a standalone transform (e.g., `t1_05_fourier_descriptors.py`).

### Backend: Other Key Modules

- `app/svg/parser.py` — Regex-based SVG parsing with order-independent attribute extraction (handles line/circle/rect/path/polyline/polygon/ellipse)
- `app/llm/enrichment_formatter.py` — Converts `PipelineContext` → enrichment text with spatial interpretation + Braille grids
- `app/llm/model_router.py` — Routes to Haiku (cheap), Sonnet (mid/frontier) via `app/config.py` settings
- `app/utils/rasterizer.py` — Unicode Braille grid rendering (U+2800-U+28FF), composite/element/group grids, silhouette descriptors
- `app/learning/` — JSONL session memory, self-reflection, knowledge tags, seed patterns
- `app/engine/resolver/` — Intent parsing + SVG generation for create/modify endpoints

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

- **Transform registration outside FastAPI:** Call `_register_transforms()` from `app.main` before running the pipeline standalone (tests do this implicitly via conftest imports)
- **Adaptive gating:** Skips T3.07/18/19/20/21 for SVGs with ≤5 elements. Tests that need these transforms must use `NO_SKIP_CONFIG` in pipeline config
- **SVG parsing:** Attribute regexes are order-independent — `cx`, `cy`, `r` can appear in any order on circle/line tags
- **Build system:** Backend uses `uv_build` (not hatchling) with `module-name="app"` and `module-root=""` in pyproject.toml. Dev deps use `[dependency-groups]` not `[project.optional-dependencies]`
- **Test fixtures:** `tests/conftest.py` provides 5 pre-parsed `PipelineContext` fixtures: `circle_ctx`, `smiley_ctx`, `home_ctx`, `bar_chart_ctx`, `settings_ctx`
- **Environment:** Backend needs `ANTHROPIC_API_KEY` in `backend/.env`. Frontend needs `BACKEND_URL` in `frontend/.env.local`

## Documentation

- `docs/vectorsight_guide.md` — Comprehensive technical guide covering all 61 transforms and enrichment format (94KB)
- `docs/prd.md` — Product requirements, demo script, research landscape
- `docs/data_spec.md` — Input/output schemas, icon sets, benchmarks
- `docs/user_journeys.md` — 5 user flows
- `docs/frontend.md` — Frontend architecture and component guide

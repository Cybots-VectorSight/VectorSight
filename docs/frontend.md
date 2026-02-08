# VectorSight Frontend Specification

**Complete frontend architecture, pages, components, and design decisions.**

---

| | |
|---|---|
| **Stack** | Next.js 16 + React 19 + Bun + Tailwind CSS 4 + Hono + Zod + TanStack Query |
| **Components** | shadcn/ui **base** only (no Radix primitives) |
| **Chat UI** | [prompt-kit](https://prompt-kit.com) (PromptInput, Message, Markdown, Loader, etc.) |
| **Deployment** | Vercel |
| **Proxy** | Frontend Hono (`/api/[[...route]]`) -> FastAPI backend (`localhost:8000`) |

---

## Table of Contents

1. [Design Decisions](#1-design-decisions)
2. [Pages Overview](#2-pages-overview)
3. [Page 1: Landing](#3-page-1-landing-)
4. [Page 2: Auth (Placeholder)](#4-page-2-auth-placeholder)
5. [Page 3: Dashboard](#5-page-3-dashboard-dashboard)
6. [Page 4: Workspace](#6-page-4-workspace-workspace)
7. [Page 5: About](#7-page-5-about-about)
8. [Page 6: Settings (Placeholder)](#8-page-6-settings-placeholder-settings)
9. [Page 7: Pricing](#9-page-7-pricing-pricing)
10. [Workspace Deep Dive](#10-workspace-deep-dive)
11. [Intent Detection](#11-intent-detection)
12. [API Integration](#12-api-integration)
13. [Key Types](#13-key-types)
14. [File Structure](#14-file-structure)
15. [Build Sequence](#15-build-sequence)
16. [Verification Checklist](#16-verification-checklist)

---

## 1. Design Decisions

### Visual Style

| Decision | Choice | Rationale |
|---|---|---|
| **Theme** | Dark + techy + clean + minimal + polished | Supabase/Linear inspired |
| **Palette** | Monochrome | No accent color. Unlike Supabase (green) or Linear (purple), VectorSight uses no standard brand color. Pure monochrome. |
| **Component Library** | shadcn/ui **base** only | Already monochrome by default, fits perfectly |
| **Chat Components** | [prompt-kit](https://prompt-kit.com) | Purpose-built AI chat UI components (PromptInput, Message, Markdown, Loader, CodeBlock, etc.) |
| **Typography** | System / Inter | Clean, minimal, techy |

### Component Rules

> **STRICT: Base UI only. No Radix.**
>
> All shadcn/ui components MUST use the **base** variant (pure HTML + Tailwind). Do NOT use
> any Radix UI primitives (`@radix-ui/*`). If a shadcn component defaults to a Radix
> primitive under the hood, either swap it for the base version or build a custom
> replacement with plain HTML + Tailwind.
>
> This keeps the bundle lean, avoids Radix's runtime overhead, and ensures full control
> over styling and behavior.

### Chat UI — prompt-kit

The entire chat experience in the workspace uses **prompt-kit** (`https://prompt-kit.com`).
Install components via the shadcn CLI pattern:

```bash
npx shadcn add "https://prompt-kit.com/c/<component>.json"
```

**prompt-kit components to use:**

| Component | Purpose | Install |
|---|---|---|
| `PromptInput` | Unified input box (auto-resize textarea + actions slot) | `prompt-input.json` |
| `Message` | Chat message bubble (user / assistant) | `message.json` |
| `Markdown` | Memoized markdown renderer (optimized for streaming) | `markdown.json` |
| `Loader` | Typing / thinking indicator while waiting for AI | `loader.json` |
| `CodeBlock` | Syntax-highlighted code blocks in responses (shiki) | `code-block.json` |
| `PromptSuggestion` | Quick-action suggestion pills | `prompt-suggestion.json` |
| `FileUpload` | SVG file drag-and-drop upload | `file-upload.json` |
| `ScrollButton` | Scroll-to-bottom button in message list | `scroll-button.json` |

### UX Decisions

| Decision | Choice | Alternatives Considered |
|---|---|---|
| **Input mode** | Unified input | Separate tabs for chat vs modify. Rejected: too many clicks. One box, auto-detect intent. |
| **Version history** | Version timeline (thumbnail strip) | Before/after split, stack with undo. Timeline chosen: most visually impressive, like Figma. |
| **Baseline comparison** | Toggle built into workspace | Separate page, curated examples only. Toggle chosen: most flexible, users can toggle both directions. |
| **Baseline direction** | Toggle both ways | One-way only. Both directions: can show enriched vs raw, or disable enrichment entirely. |
| **Layout** | 3-panel workspace | Single panel, tabs. 3-panel chosen: SVG preview + versions (left), chat (right), input (bottom-right). |
| **Navigation** | Minimal pages | Many separate pages. Kept to 7 pages total, workspace does the heavy lifting. |

---

## 2. Pages Overview

| # | Page | Route | Priority | Description |
|---|---|---|---|---|
| 1 | Landing | `/` | 2nd | Hero, hackathon info, team, CTA |
| 2 | Auth | `/login`, `/signup` | Later | Placeholder stubs, mock session |
| 3 | Dashboard | `/dashboard` | Later | Recent SVGs, saved conversations, saved icons |
| 4 | **Workspace** | `/workspace` | **1st (BUILD FIRST)** | The app. SVG preview + chat/modify + versions + baseline |
| 5 | About | `/about` | 3rd | Team, hackathon, architecture |
| 6 | Settings | `/settings` | Later | Placeholder |
| 7 | Pricing | `/pricing` | Later | Free + Enterprise, no prices shown |

---

## 3. Page 1: Landing (`/`)

### Purpose
First impression. Explain what VectorSight does, show the hackathon context, introduce Team Cybots, drive users to the workspace.

### Sections (top to bottom)

#### 3.1 Hero
- VectorSight logo (`vectorsight-light.svg` on dark background)
- One-liner: **"We transform SVG code into spatial JSON. Same model, better input, correct answers."**
- CTA button: "Try Workspace" -> `/workspace`

#### 3.2 How It Works (3-step visual)
```
SVG Code  -->  Geometry Engine  -->  LLM Reads Spatial JSON
```
- Step 1: Raw SVG code (show snippet)
- Step 2: Engine processes 61 transforms across 5 layers
- Step 3: LLM receives structured spatial data, gives correct answers

#### 3.3 Problem / Solution Comparison
Side-by-side or before/after:

| Without VectorSight | With VectorSight |
|---|---|
| LLM sees raw coordinates | LLM sees spatial relationships |
| "Maybe 8 units apart?" (wrong) | "Eyes are 5 units apart edge-to-edge" (correct) |
| Guesses at containment | Knows what's inside what |
| No spatial awareness | Full geometric understanding |

#### 3.4 Stats Banner
- **61** transforms
- **5** layers (L0-L4)
- **~1,200** tokens per enrichment
- Works with **any** LLM

#### 3.5 Hackathon Banner
- "The Strange Data Project" hackathon
- Prize pool: EUR 5,000
- Timeline: Jan 31 - Feb 15, 2025

#### 3.6 Team Cybots
- Team logo (`cybots.svg`)
- Team name and brief description

#### 3.7 Footer CTA
- "Try it now" -> `/workspace`
- Links to About, GitHub (if public)

---

## 4. Page 2: Auth (Placeholder)

### Purpose
Placeholder authentication pages. No real auth backend yet -- mock session only.

### Routes
- `/login` -- Login form stub
- `/signup` -- Signup form stub

### Behavior
- Forms render but don't connect to a real auth backend
- Mock session: clicking "Login" sets a local state/cookie
- Navigation shows "Login" when logged out, user avatar/name when "logged in"
- Placeholder for future OAuth / email auth

### Components
- Email + password form (non-functional)
- "Continue with Google" button (disabled/placeholder)
- Link between login <-> signup

---

## 5. Page 3: Dashboard (`/dashboard`)

### Purpose
User's workspace hub. View history, saved items, quick actions.

### Sections

#### 5.1 Recent SVGs
- Grid/list of recently analyzed or modified SVGs
- Thumbnail preview + name + timestamp
- Click to open in workspace

#### 5.2 Saved Conversations
- List of prior chat/modify conversations
- Preview of last message
- Click to resume in workspace

#### 5.3 Saved Icons
- All saved SVGs: originals AND modified versions
- Filter: originals, modified, all
- Download any version
- Open any version in workspace

#### 5.4 Quick Actions
- "New Analysis" -> `/workspace`
- "Upload SVG" -> `/workspace` with upload dialog
- "Try a Sample" -> `/workspace` with sample picker

#### 5.5 Navigation
- Link back to landing page (homepage)
- Link to settings
- Link to workspace

---

## 6. Page 4: Workspace (`/workspace`)

**This is the core of the application. Build this first.**

### Layout (3-panel)

```
+------------------------------------------------------------------+
|  Header                                                           |
|  [Logo]  [Landing | Workspace | About]       [Baseline Toggle]    |
+-------------------------------+----------------------------------+
|                               |                                  |
|   Left Panel                  |   Right Panel (Chat)             |
|                               |                                  |
|   +-------------------------+ |   +----------------------------+ |
|   |                         | |   |  Message List (scrollable) | |
|   |   SVG Preview           | |   |                            | |
|   |   (large, rendered)     | |   |  User: "Move eyes closer"  | |
|   |                         | |   |  AI: "Done! Moved 2 units" | |
|   |                         | |   |                            | |
|   +-------------------------+ |   |  [With VectorSight]         | |
|                               |   |  [Without VectorSight]      | |
|   +-------------------------+ |   |  (when baseline toggle ON)  | |
|   | Version Timeline        | |   +----------------------------+ |
|   | [v1] [v2] [v3] [v4]    | |                                  |
|   | (horizontal thumbnails) | |   +----------------------------+ |
|   +-------------------------+ |   | [Type message...] [Send]   | |
|                               |   +----------------------------+ |
+-------------------------------+----------------------------------+
```

### Component Tree

```
WorkspacePage
├── Header
│   ├── Logo (vectorsight-light.svg)
│   ├── Nav (Landing | Workspace | About)
│   └── BaselineToggle (switch)
├── LeftPanel
│   ├── SvgPreview (rendered SVG, large)
│   ├── SvgInput (paste/upload or sample picker, shown initially)
│   └── VersionTimeline (horizontal thumbnail strip)
│       └── VersionThumbnail[] (mini SVG previews, clickable)
└── RightPanel (chat — all prompt-kit components)
    ├── MessageList (scrollable + ScrollButton)
    │   └── Message[] (prompt-kit Message)
    │       ├── UserMessage (question or command)
    │       └── AssistantMessage
    │           ├── Markdown (prompt-kit, memoized streaming)
    │           ├── baseline Markdown (when toggle ON)
    │           └── changes list (when modify)
    ├── Loader (prompt-kit, shown while AI responds)
    ├── PromptSuggestion[] (quick actions, shown in empty state)
    └── PromptInput (prompt-kit, unified textarea + actions)
```

See [Section 10: Workspace Deep Dive](#10-workspace-deep-dive) for full details on each component.

---

## 7. Page 5: About (`/about`)

### Purpose
Team information, hackathon context, technical architecture, what VectorSight is NOT.

### Sections

#### 7.1 Team Cybots
- Team logo (`cybots.svg`)
- Team members and roles
- Brief story / motivation

#### 7.2 Hackathon Details
- "The Strange Data Project"
- EUR 5,000 prize pool
- Timeline
- What we built and why

#### 7.3 "What This Is NOT" Table

| What people think | What VectorSight actually is |
|---|---|
| Fine-tuning | No. The model stays the same. |
| RAG | No. We don't search a database. |
| Tool / MCP | No. The LLM doesn't call our code. |
| Image recognition | No. We read SVG code, not pixels. |

**It is:** Data transformation. SVG code -> spatial JSON. Same model, better input, correct answers.

#### 7.4 Architecture Diagram
- Transform layers visualization (L0 -> L1 -> L2 -> L3 -> L4)
- 61 transforms overview
- Pipeline flow: SVG -> Parser -> Engine -> Enrichment -> LLM

#### 7.5 Links
- Technical guide (`vectorsight_guide.md`)
- Data spec (`data_spec.md`)
- GitHub repository (if public)

---

## 8. Page 6: Settings (Placeholder) (`/settings`)

### Purpose
Placeholder settings page for future configuration options.

### Placeholder Sections
- Profile (name, email -- non-functional)
- Theme toggle (dark/light -- can implement dark-only for now)
- API key management (placeholder)
- Notification preferences (placeholder)
- Account deletion (placeholder)

---

## 9. Page 7: Pricing (`/pricing`)

### Purpose
Placeholder pricing page. No actual prices shown.

### Layout

```
+------------------------------------------------------------------+
|                         Pricing                                   |
|                                                                   |
|  +---------------------------+  +------------------------------+  |
|  |         Free              |  |        Enterprise            |  |
|  |                           |  |                              |  |
|  |  - Analyze SVGs           |  |  - Everything in Free        |  |
|  |  - Chat with AI           |  |  - Priority support          |  |
|  |  - Modify SVGs            |  |  - Custom integrations       |  |
|  |  - Version history        |  |  - Bulk processing           |  |
|  |  - Baseline comparison    |  |  - Team workspace            |  |
|  |                           |  |  - API access                |  |
|  |  [Get Started]            |  |  [Contact Us]                |  |
|  |                           |  |                              |  |
|  |  (no price shown)         |  |  (no price shown)            |  |
|  +---------------------------+  +------------------------------+  |
+------------------------------------------------------------------+
```

### Key Details
- Two tiers: **Free** and **Enterprise**
- NO actual prices displayed on either tier
- Free tier: all core features
- Enterprise tier: advanced/team features
- Enterprise CTA: "Contact Us" (placeholder)

---

## 10. Workspace Deep Dive

### 10.1 SVG Input (Initial State)

When workspace first loads, there's no SVG yet. Show input options:

| Method | Description | Component |
|---|---|---|
| **Paste code** | Text area for raw SVG markup | Plain `<textarea>` (base HTML) |
| **Upload file** | Drag & drop or click to select `.svg` file | prompt-kit `FileUpload` |
| **Pick sample** | Dropdown/grid of built-in test icons (smiley, house, etc.) | Custom grid (base HTML + Tailwind) |

After an SVG is loaded, the input collapses and the preview takes over. A small "Change SVG" button remains accessible.

### 10.2 SVG Preview (Left Panel, Top)

- Large rendered SVG display
- Takes up most of the left panel
- Responsive sizing
- Clean background (dark, matches theme)
- Shows the currently selected version (from timeline)

### 10.3 Version Timeline (Left Panel, Bottom)

Horizontal thumbnail strip showing SVG modification progression, like Figma's version history.

```
[Original] [Moved eyes] [Thicker strokes] [Added hat] [Current]
     ^          ^              ^                ^           ^
   v1 thumb   v2 thumb      v3 thumb        v4 thumb    v5 thumb
```

**Behavior:**
- Each version = a mini SVG thumbnail
- Click a thumbnail to view that version in the preview
- Versions are labeled: "Original", then the modification description
- Horizontal scroll if many versions
- Current version is highlighted
- Can compare any two versions (future enhancement)

### 10.4 Chat Panel (Right Panel) — prompt-kit components

Scrollable message list using prompt-kit `Message` components with `Markdown` renderer and `Loader` for thinking state. `ScrollButton` auto-appears when scrolled up.

**Message types (all rendered with prompt-kit `Message`):**

| Type | Content | Rendering |
|---|---|---|
| **User question** | The question text | `Message` (user role) |
| **User command** | The modification command text | `Message` (user role) |
| **AI answer** | Text response (for questions) | `Message` + `Markdown` (memoized, streaming-optimized) |
| **AI modification** | Response + changes list + "new version created" indicator | `Message` + `Markdown` + `CodeBlock` for SVG diff |
| **Baseline response** | Side-by-side or stacked: "With VectorSight" vs "Without VectorSight" (only when baseline toggle is ON) | Two `Markdown` blocks inside one `Message` |

**Empty state:** Show `PromptSuggestion` pills with starter prompts (e.g. "How many shapes?", "Move the eyes closer", "What colors are used?").

### 10.5 Unified Input (Right Panel, Bottom) — prompt-kit `PromptInput`

Uses prompt-kit's `PromptInput` component with `PromptInputTextarea` (auto-resize) and `PromptInputActions` (send button slot).

```
+--------------------------------------------------+--------+
| Type a question or describe a change...           | [Send] |
+--------------------------------------------------+--------+
```

**Key behavior:**
- One input box for everything (prompt-kit `PromptInput`)
- Auto-resize via `PromptInputTextarea` (maxHeight ~240px)
- `PromptInputActions` slot for send button
- Auto-detects intent (question vs command) -- see [Section 11](#11-intent-detection)
- Override prefix available: `ask:` forces chat, `edit:` forces modify
- Enter to send (`onSubmit`), Shift+Enter for newline
- Placeholder text: "Type a question or describe a change..."
- `isLoading` prop disables input while AI responds

### 10.6 Baseline Toggle (Header)

A switch/toggle in the header bar.

**When OFF (default):**
- Normal mode
- Questions go to `/api/chat` with enrichment
- Modifications go to `/api/modify` with enrichment

**When ON:**
- Comparison mode
- Each question sends TWO parallel API calls:
  1. With VectorSight enrichment (normal)
  2. Without enrichment (baseline -- raw SVG only)
- Responses display side-by-side or stacked:
  - "With VectorSight" section
  - "Without VectorSight" section
- User can visually compare quality of responses
- Toggle can be turned off at any time to return to normal mode
- Note: this doubles API calls, so it's off by default

### 10.7 Workspace State Flow

```
1. Initial state:
   - Left: SVG input (paste/upload/sample)
   - Right: Empty chat with welcome message

2. SVG loaded:
   - Left: SVG preview (large) + version timeline (just "Original")
   - Right: Chat ready, "Ask a question or describe a change"

3. User asks question:
   - Right: Message appears, loading state, AI responds
   - Left: No change (unless baseline toggle shows comparison)

4. User gives modification command:
   - Right: Message appears, loading state, AI responds with changes
   - Left: Preview updates to new version, timeline adds thumbnail

5. User clicks old version in timeline:
   - Left: Preview shows that version
   - Right: Chat unchanged (conversation continues)
```

---

## 11. Intent Detection

The unified input box auto-detects whether the user is asking a question (-> chat API) or giving a command (-> modify API).

### Detection Rules

```
INPUT -> classify:

1. Starts with "ask:" prefix     -> CHAT (forced)
2. Starts with "edit:" prefix    -> MODIFY (forced)
3. Contains question word at start:
   - what, how, where, which, is, are, do, does, can, why
   -> CHAT
4. Ends with "?"                 -> CHAT
5. Everything else               -> MODIFY
```

### Question Words (triggers CHAT)
```
what, how, where, which, is, are, do, does, can, why
```

### Override Prefixes
| Prefix | Forces | Example |
|---|---|---|
| `ask:` | Chat | `ask: make this thicker?` -> goes to chat even though it looks like a command |
| `edit:` | Modify | `edit: what if the eyes were closer` -> goes to modify even though it sounds like a question |

### API Mapping

| Intent | Endpoint | What Happens |
|---|---|---|
| CHAT | `POST /api/chat` | Sends SVG + enrichment + question. Returns text answer. |
| MODIFY | `POST /api/modify` | Sends SVG + enrichment + command. Returns new SVG + explanation. |

---

## 12. API Integration

### Proxy Architecture

```
Browser  -->  Next.js Hono proxy (/api/[[...route]])  -->  FastAPI (localhost:8000)
```

The frontend never talks directly to the FastAPI backend. All requests go through the Hono proxy in `frontend/src/app/api/[[...route]]/route.ts`.

### Backend Endpoints (7 total)

| Endpoint | Method | Purpose | Used In |
|---|---|---|---|
| `/health` | GET | Health check | App initialization |
| `/analyze` | POST | Full SVG analysis (61 transforms) | SVG load |
| `/chat` | POST | Question with enrichment context | Chat messages |
| `/modify` | POST | SVG modification via natural language | Modify commands |
| `/create` | POST | Generate new SVG from description | Future: create page |
| `/icon-set` | POST | Batch analyze icon set | Future: icon set page |
| `/playground` | POST | Raw enrichment, tweak settings | Future: playground |

### State Management

| What | How |
|---|---|
| API calls | TanStack Query (caching, loading states, error handling) |
| SVG state | React state (current SVG, version history) |
| Chat messages | React state (message array) |
| Baseline toggle | React state (boolean) |
| Form inputs | React state / controlled components |
| Schema validation | Zod |

### Dependency Rules

> **No `@radix-ui/*` packages.** If `npx shadcn add` tries to pull in a Radix
> dependency, use the **base** variant instead or build a plain HTML + Tailwind
> replacement. Run `bun pm ls | grep radix` periodically to verify zero Radix deps.
>
> **prompt-kit is the only chat UI library.** Do not add `@ai-sdk/react`, Vercel AI SDK
> chat hooks, or any other chat component library. prompt-kit components are installed
> standalone via its shadcn-compatible registry.

---

## 13. Key Types

```typescript
/** A single version in the SVG modification timeline */
type SvgVersion = {
  id: string
  svg: string           // raw SVG code
  label: string         // "Original" | "Moved eyes closer" | ...
  timestamp: number
}

/** A message in the chat panel */
type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  intent: "chat" | "modify"
  baselineContent?: string    // response without enrichment (when toggle ON)
  svgVersionId?: string       // links to version created (for modify)
  timestamp: number
}

/** Workspace state */
type WorkspaceState = {
  currentSvg: string | null
  versions: SvgVersion[]
  activeVersionId: string | null
  messages: ChatMessage[]
  baselineEnabled: boolean
  isLoading: boolean
}
```

---

## 14. File Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx                    # Update: add Header, dark theme, font
│   │   ├── page.tsx                      # Rewrite: landing page
│   │   ├── workspace/
│   │   │   └── page.tsx                  # NEW: workspace page (BUILD FIRST)
│   │   ├── about/
│   │   │   └── page.tsx                  # NEW: about page
│   │   ├── login/
│   │   │   └── page.tsx                  # NEW: auth placeholder
│   │   ├── signup/
│   │   │   └── page.tsx                  # NEW: auth placeholder
│   │   ├── dashboard/
│   │   │   └── page.tsx                  # NEW: dashboard
│   │   ├── settings/
│   │   │   └── page.tsx                  # NEW: settings placeholder
│   │   ├── pricing/
│   │   │   └── page.tsx                  # NEW: pricing page
│   │   └── api/[[...route]]/
│   │       └── route.ts                  # Update: add proxy routes to FastAPI
│   ├── components/
│   │   ├── header.tsx                    # Site header with nav + baseline toggle
│   │   ├── svg-preview.tsx               # Large rendered SVG display
│   │   ├── svg-input.tsx                 # Paste/upload/sample picker (prompt-kit FileUpload)
│   │   ├── chat-panel.tsx                # Chat container (message list + input)
│   │   ├── version-timeline.tsx          # Horizontal thumbnail strip
│   │   ├── version-thumbnail.tsx         # Mini SVG preview (clickable)
│   │   ├── baseline-toggle.tsx           # Switch for baseline comparison
│   │   ├── prompt-kit/                   # prompt-kit components (installed via CLI)
│   │   │   ├── prompt-input.tsx          # Unified textarea + actions
│   │   │   ├── message.tsx              # Chat message bubble
│   │   │   ├── markdown.tsx             # Memoized markdown renderer
│   │   │   ├── loader.tsx               # AI thinking indicator
│   │   │   ├── code-block.tsx           # Syntax-highlighted code (shiki)
│   │   │   ├── prompt-suggestion.tsx    # Quick-action suggestion pills
│   │   │   ├── file-upload.tsx          # Drag-and-drop file upload
│   │   │   └── scroll-button.tsx        # Scroll-to-bottom button
│   │   └── ui/                           # shadcn/ui BASE components (NO Radix)
│   │       ├── button.tsx               # base variant
│   │       ├── input.tsx                # base variant
│   │       ├── switch.tsx               # base variant (HTML checkbox, no @radix-ui/switch)
│   │       ├── card.tsx                 # base variant
│   │       └── ...                      # all base — never @radix-ui/*
│   ├── lib/
│   │   ├── api.ts                        # API client (TanStack Query hooks)
│   │   ├── intent.ts                     # Intent detection logic
│   │   └── types.ts                      # TypeScript types (SvgVersion, ChatMessage, etc.)
│   └── styles/
│       └── globals.css                   # Tailwind base + dark theme
├── public/
│   ├── vectorsight-light.svg             # Logo for dark backgrounds
│   ├── vectorsight-dark.svg              # Logo for light backgrounds
│   └── cybots.svg                        # Team logo
├── components.json                       # shadcn/ui config
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── vercel.json                           # Vercel deployment config (Bun)
```

---

## 15. Build Sequence

| Step | What | Details |
|---|---|---|
| **1** | **Workspace page** | SVG input + preview + empty chat panel. Get the layout working. |
| **2** | **API proxy** | Wire up Hono proxy to forward to FastAPI backend. Test with `/health`. |
| **3** | **Chat flow** | Unified input -> intent detection -> API call -> display response. |
| **4** | **Modify flow** | Detect modify intent -> call `/modify` -> update SVG preview -> add to version timeline. |
| **5** | **Version timeline** | Horizontal thumbnail strip. Click to switch. |
| **6** | **Baseline toggle** | Switch in header. When ON, parallel API calls. Side-by-side/stacked responses. |
| **7** | **Landing page** | Hero, how it works, problem/solution, stats, hackathon, team, CTA. |
| **8** | **About page** | Team, hackathon, "what this is NOT", architecture. |
| **9** | **Auth stubs** | Login/signup placeholder pages, mock session. |
| **10** | **Dashboard** | Recent SVGs, saved conversations, saved icons. |
| **11** | **Settings** | Placeholder settings page. |
| **12** | **Pricing** | Free + Enterprise tiers, no prices. |

**Priority: Steps 1-6 are the MVP. Steps 7-8 for hackathon submission. Steps 9-12 are polish.**

---

## 16. Verification Checklist

### Workspace MVP
- [ ] SVG can be loaded via paste, upload, or sample
- [ ] SVG renders correctly in preview panel
- [ ] Unified input accepts text and sends on Enter
- [ ] Intent detection correctly classifies questions vs commands
- [ ] Chat messages display in scrollable list
- [ ] Modify commands update the SVG preview
- [ ] Version timeline shows all SVG versions as thumbnails
- [ ] Clicking a version thumbnail switches the preview
- [ ] Baseline toggle sends parallel API calls
- [ ] Baseline responses show side-by-side comparison

### Landing Page
- [ ] Hero with logo, one-liner, CTA
- [ ] How it works section (3 steps)
- [ ] Problem/solution comparison
- [ ] Stats banner
- [ ] Hackathon info
- [ ] Team Cybots section
- [ ] CTA links to `/workspace`

### Navigation
- [ ] Header appears on all pages
- [ ] Logo links to landing page
- [ ] Nav links work (Landing, Workspace, About)
- [ ] Baseline toggle only visible on workspace
- [ ] Dashboard links back to landing/homepage

### Polish Pages
- [ ] Auth pages render (non-functional)
- [ ] Dashboard shows placeholder content
- [ ] Settings page renders
- [ ] Pricing page shows two tiers, no prices
- [ ] All pages respect dark monochrome theme

---

**Next:** See `user_journeys.md` for detailed interaction flows per feature, and `data_spec.md` for API request/response schemas.

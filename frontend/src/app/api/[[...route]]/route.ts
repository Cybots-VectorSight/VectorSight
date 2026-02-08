import { Hono } from "hono"
import { handle } from "hono/vercel"

type StatusCode = 200 | 400 | 401 | 403 | 404 | 422 | 500 | 502 | 503

const app = new Hono().basePath("/api")

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

async function proxyToBackend(
  path: string,
  method: string,
  body?: unknown
): Promise<{ data: unknown; status: StatusCode }> {
  const url = `${BACKEND_URL}/api${path}`
  const res = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await res.json()
  return { data, status: (res.ok ? 200 : res.status) as StatusCode }
}

// Health check
app.get("/health", async (c) => {
  try {
    const { data, status } = await proxyToBackend("/health", "GET")
    return c.json(data, status)
  } catch {
    return c.json({ status: "error", message: "Backend unreachable" }, 503)
  }
})

// Analyze SVG
app.post("/analyze", async (c) => {
  try {
    const body = await c.req.json()
    const { data, status } = await proxyToBackend("/analyze", "POST", body)
    return c.json(data, status)
  } catch {
    return c.json({ error: "Analysis failed" }, 500)
  }
})

// Chat about SVG (non-streaming)
app.post("/chat", async (c) => {
  try {
    const body = await c.req.json()
    const { data, status } = await proxyToBackend("/chat", "POST", body)
    return c.json(data, status)
  } catch {
    return c.json({ error: "Chat failed" }, 500)
  }
})

// Chat streaming (SSE passthrough)
app.post("/chat/stream", async (c) => {
  try {
    const body = await c.req.json()
    const url = `${BACKEND_URL}/api/chat/stream`
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })

    if (!res.ok || !res.body) {
      return c.json({ error: "Stream failed" }, res.status as StatusCode)
    }

    return new Response(res.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    })
  } catch {
    return c.json({ error: "Stream failed" }, 500)
  }
})

// Modify SVG
app.post("/modify", async (c) => {
  try {
    const body = await c.req.json()
    const { data, status } = await proxyToBackend("/modify", "POST", body)
    return c.json(data, status)
  } catch {
    return c.json({ error: "Modification failed" }, 500)
  }
})

// Create SVG
app.post("/create", async (c) => {
  try {
    const body = await c.req.json()
    const { data, status } = await proxyToBackend("/create", "POST", body)
    return c.json(data, status)
  } catch {
    return c.json({ error: "Creation failed" }, 500)
  }
})

export const GET = handle(app)
export const POST = handle(app)
export const PUT = handle(app)
export const DELETE = handle(app)

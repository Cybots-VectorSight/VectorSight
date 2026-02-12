/** Parse SSE text into events, preserving state across calls via currentEvent */
export function parseSSE(
  buffer: string,
  currentEvent = "",
): { events: Array<{ event: string; data: string }>; remaining: string; currentEvent: string } {
  const events: Array<{ event: string; data: string }> = []
  const lines = buffer.split("\n")
  let remaining = ""

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // If it's the last line and doesn't end with \n, it's incomplete
    if (i === lines.length - 1 && !buffer.endsWith("\n")) {
      remaining = line
      break
    }

    if (line.startsWith("event: ")) {
      currentEvent = line.slice(7).trim()
    } else if (line.startsWith("data: ")) {
      const data = line.slice(6)
      events.push({ event: currentEvent || "message", data })
      currentEvent = ""
    } else if (line === "") {
      // Empty line = end of event (already pushed)
    }
  }

  return { events, remaining, currentEvent }
}

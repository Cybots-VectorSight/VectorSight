/**
 * Client-side SVG anonymizer — strips colors, IDs, classes, styles, metadata.
 * Keeps only geometry + transforms. TypeScript port of svg_anonymizer.py.
 */

const REMOVE_TAGS = new Set([
  "title",
  "desc",
  "metadata",
  "style",
  "defs",
  "namedview",
  "perspective",
])

const KEEP_ATTRS: Record<string, string[]> = {
  path: ["d", "transform"],
  circle: ["cx", "cy", "r", "transform"],
  ellipse: ["cx", "cy", "rx", "ry", "transform"],
  rect: ["x", "y", "width", "height", "rx", "ry", "transform"],
  line: ["x1", "y1", "x2", "y2", "transform"],
  polyline: ["points", "transform"],
  polygon: ["points", "transform"],
}

const SHAPE_TAGS = new Set(Object.keys(KEEP_ATTRS))

interface Shape {
  type: string
  [key: string]: string
}

function stripNs(tag: string): string {
  const i = tag.lastIndexOf(":")
  return i >= 0 ? tag.slice(i + 1) : tag
}

function extractShapes(el: Element): Shape[] {
  const shapes: Shape[] = []
  const tag = stripNs(el.tagName.toLowerCase())

  if (REMOVE_TAGS.has(tag)) return shapes

  if (SHAPE_TAGS.has(tag)) {
    const kept: Shape = { type: tag }
    for (const attr of KEEP_ATTRS[tag]) {
      const val = el.getAttribute(attr)
      if (val) kept[attr] = val
    }
    if (tag === "path" && !kept.d) {
      // skip paths without d
    } else {
      shapes.push(kept)
    }
  }

  for (const child of Array.from(el.children)) {
    shapes.push(...extractShapes(child))
  }

  return shapes
}

function getViewBox(svg: Element): string {
  const vb = svg.getAttribute("viewBox")
  if (vb) return vb
  const w = (svg.getAttribute("width") ?? "100").replace(/px|pt/g, "")
  const h = (svg.getAttribute("height") ?? "100").replace(/px|pt/g, "")
  const wn = parseFloat(w) || 100
  const hn = parseFloat(h) || 100
  return `0 0 ${wn} ${hn}`
}

export interface AnonymizeResult {
  cleanSvg: string
  claudeText: string
  summary: string
}

export function anonymizeSvg(svgText: string): AnonymizeResult | null {
  // Strip XML comments
  const cleaned = svgText.replace(/<!--[\s\S]*?-->/g, "")

  const parser = new DOMParser()
  const doc = parser.parseFromString(cleaned, "image/svg+xml")
  const root = doc.documentElement

  if (root.tagName.toLowerCase() !== "svg") return null

  const parseError = root.querySelector("parsererror")
  if (parseError) return null

  const viewbox = getViewBox(root)
  const shapes = extractShapes(root)

  if (shapes.length === 0) return null

  // Build clean SVG
  const lines = [`<svg viewBox="${viewbox}" xmlns="http://www.w3.org/2000/svg">`]
  for (const s of shapes) {
    const attrs = Object.entries(s)
      .filter(([k]) => k !== "type")
      .map(([k, v]) => `${k}="${v}"`)
      .join(" ")
    lines.push(`  <${s.type} ${attrs} />`)
  }
  lines.push("</svg>")
  const cleanSvg = lines.join("\n")

  // Build Claude-ready text
  const claudeLines = [`viewBox: ${viewbox}`, `Elements: ${shapes.length}`, ""]
  for (let i = 0; i < shapes.length; i++) {
    const s = shapes[i]
    let extra = ""
    if (s.type === "path" && s.d) {
      const subPaths = (s.d.match(/[Mm]/g) ?? []).length
      extra = `, ${subPaths} sub-path${subPaths !== 1 ? "s" : ""}`
    }
    claudeLines.push(`--- Element ${i} (${s.type}${extra}) ---`)
    if (s.type === "path") {
      claudeLines.push(s.d ?? "")
      if (s.transform) claudeLines.push(`transform: ${s.transform}`)
    } else {
      const attrs = Object.entries(s)
        .filter(([k]) => k !== "type")
        .reduce(
          (acc, [k, v]) => {
            acc[k] = v
            return acc
          },
          {} as Record<string, string>
        )
      claudeLines.push(JSON.stringify(attrs))
    }
    claudeLines.push("")
  }
  const claudeText = claudeLines.join("\n")

  // Summary
  const typeCounts: Record<string, number> = {}
  for (const s of shapes) {
    typeCounts[s.type] = (typeCounts[s.type] ?? 0) + 1
  }
  const parts = Object.entries(typeCounts)
    .map(([k, v]) => `${v} ${k}${v > 1 ? "s" : ""}`)
    .join(", ")
  const summary = `viewBox: ${viewbox} | ${shapes.length} elements (${parts})`

  return { cleanSvg, claudeText, summary }
}

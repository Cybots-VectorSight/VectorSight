import { z } from "zod/v4"

// --- Core types ---

export interface SvgVersion {
  id: string
  svg: string
  label: string
  timestamp: number
}

export type MessageRole = "user" | "assistant"
export type Intent = "chat" | "modify"

export interface EditOp {
  action: "add" | "delete" | "modify"
  target?: string
  position?: string
  svg_fragment?: string
  attributes?: Record<string, string>
}

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  reasoning?: string
  intent?: Intent
  baselineContent?: string
  svgVersionId?: string
  svgChanges?: string[]
  editOps?: EditOp[]
  editReasoning?: string
  timestamp: number
}

export interface WorkspaceState {
  currentSvg: string | null
  versions: SvgVersion[]
  activeVersionId: string | null
  messages: ChatMessage[]
  baselineEnabled: boolean
  isLoading: boolean
}

// --- API schemas ---

export const AnalyzeRequestSchema = z.object({
  svg: z.string(),
})

export const ChatRequestSchema = z.object({
  svg: z.string(),
  question: z.string(),
  history: z.array(z.record(z.string(), z.string())).optional(),
})

export const ModifyRequestSchema = z.object({
  svg: z.string(),
  instruction: z.string(),
})

export const CreateRequestSchema = z.object({
  description: z.string(),
})

// --- API response types ---

export interface ElementSummary {
  id: string
  shape_class: string
  area: number
  bbox: [number, number, number, number]
  centroid: [number, number]
  circularity: number
  convexity: number
  aspect_ratio: number
  size_tier: string
}

export interface SymmetryInfo {
  axis_type: string
  score: number
  pairs: [string, string][]
  on_axis: string[]
  rotational_order?: number
}

export interface AnalyzeResponse {
  enrichment: {
    source: string
    canvas: [number, number]
    element_count: number
    subpath_count: number
    is_stroke_based: boolean
    elements: ElementSummary[]
    symmetry: SymmetryInfo
    size_tiers: Record<string, string[]>
    ascii_grid_positive: string
    ascii_grid_negative: string
    enrichment_text: string
  }
  processing_time_ms: number
  transforms_completed: number
  transforms_failed: number
  errors: Record<string, string>
}

export interface ChatResponse {
  answer: string
  enrichment_used: boolean
}

export interface ModifyResponse {
  svg: string
  changes: string[]
  edit_ops?: EditOp[]
  reasoning?: string
}

export interface CreateResponse {
  svg: string
  intent: string
  validation_passed: boolean
}

export interface TransformProgress {
  transform_id: string
  description: string
  layer: string
  layer_index: number
  index: number
  total: number
  elapsed_ms: number
  status: "ok" | "error" | "running"
  error: string
  sub_progress?: number  // 0.0–1.0 for long transforms (e.g. T3.01)
}

export interface StreamingAnalyzeResult extends AnalyzeResponse {
  estimated_tokens: number
}

export interface HealthResponse {
  status: string
  version: string
  transforms_registered: number
}

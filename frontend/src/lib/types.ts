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

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  reasoning?: string
  intent?: Intent
  baselineContent?: string
  svgVersionId?: string
  svgChanges?: string[]
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

export interface AnalyzeResponse {
  enrichment: Record<string, unknown>
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
}

export interface CreateResponse {
  svg: string
  intent: string
  validation_passed: boolean
}

export interface HealthResponse {
  status: string
  version: string
  transforms_registered: number
}

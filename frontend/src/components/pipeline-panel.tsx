"use client"

import { useState } from "react"
import { AlertTriangle } from "lucide-react"
import type {
  StreamingAnalyzeResult,
  StepVisual,
} from "@/lib/types"

type SubTab = "context" | "visuals"

interface PipelinePanelProps {
  data: StreamingAnalyzeResult | null
  transforms: unknown[]
  stepVisuals: StepVisual[]
}

export function PipelinePanel({ data, stepVisuals }: PipelinePanelProps) {
  const [subTab, setSubTab] = useState<SubTab>("context")

  if (!data) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <p className="text-sm text-muted-foreground">
          Load an SVG to inspect the pipeline
        </p>
      </div>
    )
  }

  const { enrichment, estimated_tokens } = data
  const errorCount = Object.keys(data.errors).length
  const totalMs = Math.round(data.processing_time_ms)

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header stats */}
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <Badge>
          {enrichment.canvas?.[0]}x{enrichment.canvas?.[1]}
        </Badge>
        <Badge>{enrichment.element_count} groups</Badge>
        <Badge>{enrichment.subpath_count} subpaths</Badge>
        <Badge>{totalMs}ms</Badge>
        <Badge>~{estimated_tokens.toLocaleString()} tokens</Badge>
        {errorCount > 0 && (
          <Badge className="bg-destructive/10 text-destructive">
            {errorCount} failed
          </Badge>
        )}
      </div>

      {/* Tab bar */}
      <div className="flex shrink-0 items-stretch border-b border-border bg-muted/30">
        {(
          [
            ["context", "Context"],
            ["visuals", `Visuals${stepVisuals.length > 0 ? ` (${stepVisuals.length})` : ""}`],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            onClick={() => setSubTab(id as SubTab)}
            className={`px-4 py-1.5 text-[11px] font-medium transition-colors ${
              subTab === id
                ? "border-b-2 border-foreground -mb-px text-foreground bg-background"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {subTab === "context" && (
          <ContextTab
            enrichmentText={enrichment.enrichment_text}
            errors={data.errors}
            errorCount={errorCount}
          />
        )}
        {subTab === "visuals" && <VisualsTab stepVisuals={stepVisuals} />}
      </div>
    </div>
  )
}

// --- Context tab ---

function ContextTab({
  enrichmentText,
  errors,
  errorCount,
}: {
  enrichmentText: string
  errors: Record<string, string>
  errorCount: number
}) {
  return (
    <div className="p-4">
      {errorCount > 0 && (
        <div className="mb-3 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2">
          <div className="flex items-center gap-1.5 text-[11px] font-medium text-destructive">
            <AlertTriangle className="h-3 w-3" />
            {errorCount} pipeline {errorCount === 1 ? "error" : "errors"}
          </div>
          <div className="mt-1 space-y-0.5">
            {Object.entries(errors).map(([step, msg]) => (
              <p key={step} className="text-[11px] text-muted-foreground">
                <span className="font-mono text-destructive">{step}</span>: {msg}
              </p>
            ))}
          </div>
        </div>
      )}

      <pre className="overflow-auto whitespace-pre-wrap rounded border border-border bg-card p-4 font-mono text-[11px] leading-relaxed text-muted-foreground">
        {enrichmentText || "No enrichment text generated"}
      </pre>
    </div>
  )
}

// --- Visuals tab ---

function VisualsTab({ stepVisuals }: { stepVisuals: StepVisual[] }) {
  if (stepVisuals.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <p className="text-sm text-muted-foreground">
          No step visuals available
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4 p-4">
      {stepVisuals.map((sv) => (
        <div
          key={sv.transform_id}
          className="overflow-hidden rounded-lg border border-border"
        >
          <div className="flex items-center gap-2 border-b border-border bg-muted/30 px-3 py-1.5">
            <span className="font-mono text-[11px] text-muted-foreground">
              {sv.transform_id}
            </span>
            <span className="text-[11px] font-medium">{sv.label}</span>
          </div>
          <div
            className="flex min-h-[180px] items-center justify-center overflow-hidden bg-[#1a1a2e]"
            dangerouslySetInnerHTML={{ __html: sv.svg }}
          />
        </div>
      ))}
    </div>
  )
}

// --- Badge ---

function Badge({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <span
      className={`inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground ${className ?? ""}`}
    >
      {children}
    </span>
  )
}

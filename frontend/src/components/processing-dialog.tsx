"use client"

import { useEffect, useRef } from "react"
import { Check, X } from "lucide-react"
import { Loader } from "@/components/prompt-kit/loader"
import { Button } from "@/components/ui/button"
import type { TransformProgress, StreamingAnalyzeResult } from "@/lib/types"

const LAYER_LABELS: Record<string, string> = {
  PARSING: "L0 Parsing",
  SHAPE_ANALYSIS: "L1 Shape Analysis",
  VISUALIZATION: "L2 Visualization",
  RELATIONSHIPS: "L3 Relationships",
  VALIDATION: "L4 Validation",
}

type Phase = "streaming" | "complete" | "error"

interface ProcessingDialogProps {
  phase: Phase
  transforms: TransformProgress[]
  result: StreamingAnalyzeResult | null
  error: string | null
  onClose: () => void
}

export function ProcessingDialog({
  phase,
  transforms,
  result,
  error,
  onClose,
}: ProcessingDialogProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (phase === "streaming" && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [transforms, phase])

  const completed = transforms.filter((t) => t.status !== "running").length
  const total = transforms[0]?.total ?? 0
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0
  const currentLayer = transforms[transforms.length - 1]?.layer
  const layerLabel = currentLayer
    ? LAYER_LABELS[currentLayer] ?? currentLayer
    : ""
  const errorCount = transforms.filter((t) => t.status === "error").length

  // Group transforms by layer
  const byLayer = new Map<string, TransformProgress[]>()
  for (const t of transforms) {
    const group = byLayer.get(t.layer) ?? []
    group.push(t)
    byLayer.set(t.layer, group)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="mx-4 flex max-h-[80vh] w-full max-w-lg flex-col overflow-hidden rounded-xl border border-border bg-background shadow-2xl">
        {/* Header */}
        <div className="shrink-0 border-b border-border px-5 py-4">
          <h2 className="text-sm font-semibold">
            {phase === "streaming"
              ? "Processing SVG"
              : phase === "error"
                ? "Processing Failed"
                : "Processing Complete"}
          </h2>
          {phase === "streaming" && (
            <div className="mt-3">
              <div className="mb-1.5 flex items-center justify-between text-xs">
                <span className="font-medium">
                  {completed}/{total} — {layerLabel}
                </span>
                <span className="text-muted-foreground">{pct}%</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-150"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )}
          {phase === "complete" && result && (
            <div className="mt-2 flex flex-wrap gap-2">
              <Badge>{completed} transforms</Badge>
              <Badge>{Math.round(result.processing_time_ms)}ms</Badge>
              <Badge>~{result.estimated_tokens.toLocaleString()} tokens</Badge>
              {errorCount > 0 && (
                <Badge className="bg-destructive/10 text-destructive">
                  {errorCount} failed
                </Badge>
              )}
            </div>
          )}
        </div>

        {/* Transform list */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-3">
          {phase === "error" && (
            <p className="text-sm text-destructive">{error}</p>
          )}

          {Array.from(byLayer.entries()).map(([layer, items]) => (
            <div key={layer} className="mb-3">
              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                {LAYER_LABELS[layer] ?? layer}
              </p>
              {items.map((t) => (
                <div
                  key={t.transform_id}
                  className={`flex items-center gap-2 py-0.5 text-xs ${t.status === "running" ? "text-foreground" : ""}`}
                >
                  {t.status === "ok" ? (
                    <Check className="h-3 w-3 shrink-0 text-emerald-500" />
                  ) : t.status === "running" ? (
                    <Loader variant="circular" size="sm" className="h-3 w-3 shrink-0" />
                  ) : (
                    <X className="h-3 w-3 shrink-0 text-destructive" />
                  )}
                  <span className="font-mono text-muted-foreground">
                    {t.transform_id}
                  </span>
                  <span className="truncate">
                    {t.description || t.transform_id}
                  </span>
                  {t.status === "running" ? (
                    <span className="ml-auto shrink-0 text-muted-foreground animate-pulse">
                      {t.sub_progress != null
                        ? `${Math.round(t.sub_progress * 100)}%`
                        : "running..."}
                    </span>
                  ) : (
                    <span className="ml-auto shrink-0 text-muted-foreground">
                      {t.elapsed_ms.toFixed(0)}ms
                    </span>
                  )}
                </div>
              ))}
            </div>
          ))}

          {phase === "streaming" && (
            <div className="flex items-center gap-2 py-2">
              <Loader variant="circular" size="sm" />
              <span className="text-xs text-muted-foreground">
                Processing...
              </span>
            </div>
          )}
        </div>

        {/* Footer */}
        {phase !== "streaming" && (
          <div className="shrink-0 border-t border-border px-5 py-3">
            <Button size="sm" onClick={onClose} className="w-full">
              {phase === "error" ? "Dismiss" : "Close"}
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}

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

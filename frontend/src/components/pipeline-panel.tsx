"use client"

import { useState, useEffect } from "react"
import { Check, X } from "lucide-react"
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordion"
import type {
  ElementSummary,
  TransformProgress,
  StreamingAnalyzeResult,
} from "@/lib/types"

type SubTab = "results" | "enrichment" | "prompts"

interface PipelinePanelProps {
  data: StreamingAnalyzeResult | null
  transforms: TransformProgress[]
}

const PROMPT_LABELS: Record<string, string> = {
  chat: "Chat",
  modify: "Modify",
  edit: "Edit (surgical)",
  create: "Create",
  icon_set: "Icon Set",
  playground: "Playground",
}

export function PipelinePanel({ data, transforms }: PipelinePanelProps) {
  const [subTab, setSubTab] = useState<SubTab>("results")

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
  const transformCount =
    transforms.length > 0 ? transforms.length : data.transforms_completed

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header badges */}
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <Badge>{transformCount} transforms</Badge>
        <Badge>{totalMs}ms</Badge>
        <Badge>~{estimated_tokens.toLocaleString()} tokens</Badge>
        {errorCount > 0 && (
          <Badge className="bg-destructive/10 text-destructive">
            {errorCount} failed
          </Badge>
        )}
      </div>

      {/* Sub-tab bar */}
      <div className="flex shrink-0 items-stretch border-b border-border bg-muted/30">
        {(
          [
            ["results", "Results"],
            ["enrichment", "Enrichment"],
            ["prompts", "Prompts"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            onClick={() => setSubTab(id)}
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

      {/* Sub-tab content */}
      <div className="flex-1 overflow-y-auto">
        {subTab === "results" && (
          <ResultsTab
            data={data}
            transforms={transforms}
            enrichment={enrichment}
            errorCount={errorCount}
          />
        )}
        {subTab === "enrichment" && (
          <EnrichmentTab enrichmentText={enrichment.enrichment_text} />
        )}
        {subTab === "prompts" && <PromptsTab />}
      </div>
    </div>
  )
}

// --- Results sub-tab ---

function ResultsTab({
  data,
  transforms,
  enrichment,
  errorCount,
}: {
  data: StreamingAnalyzeResult
  transforms: TransformProgress[]
  enrichment: StreamingAnalyzeResult["enrichment"]
  errorCount: number
}) {
  return (
    <div className="px-4">
      <Accordion multiple defaultValue={["enrichment-text"]}>
        <AccordionItem value="enrichment-text">
          <AccordionTrigger>Enrichment Text</AccordionTrigger>
          <AccordionContent>
            <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded bg-card p-3 font-mono text-[11px] text-muted-foreground">
              {enrichment.enrichment_text || "No enrichment text generated"}
            </pre>
          </AccordionContent>
        </AccordionItem>

        {transforms.length > 0 && (
          <AccordionItem value="transform-log">
            <AccordionTrigger>
              Transform Log ({transforms.length})
            </AccordionTrigger>
            <AccordionContent>
              <div className="max-h-64 overflow-y-auto">
                {transforms.map((t) => (
                  <div
                    key={t.transform_id}
                    className="flex items-center gap-2 py-0.5 text-xs"
                  >
                    {t.status === "ok" ? (
                      <Check className="h-3 w-3 shrink-0 text-emerald-500" />
                    ) : (
                      <X className="h-3 w-3 shrink-0 text-destructive" />
                    )}
                    <span className="font-mono text-muted-foreground">
                      {t.transform_id}
                    </span>
                    <span className="truncate">
                      {t.description || t.transform_id}
                    </span>
                    <span className="ml-auto shrink-0 text-muted-foreground">
                      {t.elapsed_ms.toFixed(0)}ms
                    </span>
                  </div>
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        )}

        <AccordionItem value="elements">
          <AccordionTrigger>
            Elements ({enrichment.elements?.length ?? 0})
          </AccordionTrigger>
          <AccordionContent>
            {enrichment.elements && enrichment.elements.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-1 pr-3 font-medium">ID</th>
                      <th className="pb-1 pr-3 font-medium">Shape</th>
                      <th className="pb-1 pr-3 font-medium">Size</th>
                      <th className="pb-1 pr-3 font-medium text-right">
                        Area
                      </th>
                      <th className="pb-1 pr-3 font-medium text-right">
                        Circ.
                      </th>
                      <th className="pb-1 font-medium text-right">AR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {enrichment.elements.map((el: ElementSummary) => (
                      <tr key={el.id} className="border-b border-border/50">
                        <td className="py-1 pr-3 font-mono">{el.id}</td>
                        <td className="py-1 pr-3">{el.shape_class}</td>
                        <td className="py-1 pr-3">{el.size_tier}</td>
                        <td className="py-1 pr-3 text-right font-mono">
                          {el.area.toFixed(1)}
                        </td>
                        <td className="py-1 pr-3 text-right font-mono">
                          {el.circularity.toFixed(2)}
                        </td>
                        <td className="py-1 text-right font-mono">
                          {el.aspect_ratio.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No elements</p>
            )}
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="symmetry">
          <AccordionTrigger>Symmetry</AccordionTrigger>
          <AccordionContent>
            {enrichment.symmetry ? (
              <div className="space-y-1 text-xs">
                <p>
                  <span className="text-muted-foreground">Axis:</span>{" "}
                  {enrichment.symmetry.axis_type}
                </p>
                <p>
                  <span className="text-muted-foreground">Score:</span>{" "}
                  {enrichment.symmetry.score.toFixed(3)}
                </p>
                {enrichment.symmetry.pairs.length > 0 && (
                  <p>
                    <span className="text-muted-foreground">Pairs:</span>{" "}
                    {enrichment.symmetry.pairs
                      .map(([a, b]) => `${a}-${b}`)
                      .join(", ")}
                  </p>
                )}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No symmetry data</p>
            )}
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="canvas">
          <AccordionTrigger>Canvas</AccordionTrigger>
          <AccordionContent>
            <div className="space-y-1 text-xs">
              <p>
                <span className="text-muted-foreground">Size:</span>{" "}
                {enrichment.canvas?.[0]} x {enrichment.canvas?.[1]}
              </p>
              <p>
                <span className="text-muted-foreground">Type:</span>{" "}
                {enrichment.is_stroke_based ? "stroke-based" : "fill-based"}
              </p>
              <p>
                <span className="text-muted-foreground">Elements:</span>{" "}
                {enrichment.element_count}
              </p>
              <p>
                <span className="text-muted-foreground">Subpaths:</span>{" "}
                {enrichment.subpath_count}
              </p>
            </div>
          </AccordionContent>
        </AccordionItem>

        {enrichment.ascii_grid_positive && (
          <AccordionItem value="ascii-grid">
            <AccordionTrigger>ASCII Grid</AccordionTrigger>
            <AccordionContent>
              <pre className="max-h-64 overflow-auto rounded bg-card p-3 font-mono text-[10px] leading-none text-muted-foreground">
                {enrichment.ascii_grid_positive}
              </pre>
            </AccordionContent>
          </AccordionItem>
        )}

        {errorCount > 0 && (
          <AccordionItem value="errors">
            <AccordionTrigger className="text-destructive">
              Errors ({errorCount})
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-1">
                {Object.entries(data.errors).map(([transform, msg]) => (
                  <div key={transform} className="text-xs">
                    <span className="font-mono text-destructive">
                      {transform}
                    </span>
                    <span className="text-muted-foreground">: {msg}</span>
                  </div>
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        )}
      </Accordion>
    </div>
  )
}

// --- Enrichment sub-tab ---

function EnrichmentTab({ enrichmentText }: { enrichmentText: string }) {
  return (
    <div className="p-4">
      <p className="mb-2 text-[11px] font-medium text-muted-foreground">
        Raw enrichment text injected into the LLM system prompt as{" "}
        <code className="rounded bg-muted px-1">{"{enrichment}"}</code>
      </p>
      <pre className="overflow-auto whitespace-pre-wrap rounded border border-border bg-card p-4 font-mono text-[11px] leading-relaxed text-muted-foreground">
        {enrichmentText || "No enrichment text generated"}
      </pre>
    </div>
  )
}

// --- Prompts sub-tab (fetched from backend API) ---

function PromptsTab() {
  const [prompts, setPrompts] = useState<Record<string, string> | null>(null)
  const [activeKey, setActiveKey] = useState("chat")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch("/api/prompts")
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status}`)
        return res.json()
      })
      .then((data) => setPrompts(data))
      .catch((err) => setError(err.message))
  }, [])

  if (error) {
    return (
      <div className="p-4 text-xs text-destructive">
        Failed to load prompts: {error}
      </div>
    )
  }

  if (!prompts) {
    return (
      <div className="flex items-center justify-center p-8">
        <p className="text-xs text-muted-foreground">Loading prompts...</p>
      </div>
    )
  }

  const keys = Object.keys(prompts)

  return (
    <div className="flex h-full flex-col">
      {/* Prompt selector */}
      <div className="flex shrink-0 flex-wrap gap-1 border-b border-border px-4 py-2">
        {keys.map((key) => (
          <button
            key={key}
            onClick={() => setActiveKey(key)}
            className={`rounded px-2.5 py-1 text-[11px] font-medium transition-colors ${
              activeKey === key
                ? "bg-foreground text-background"
                : "bg-muted text-muted-foreground hover:text-foreground"
            }`}
          >
            {PROMPT_LABELS[key] ?? key}
          </button>
        ))}
      </div>

      {/* Prompt content */}
      <div className="flex-1 overflow-y-auto p-4">
        <p className="mb-2 text-[11px] font-medium text-muted-foreground">
          Live from backend.{" "}
          <code className="rounded bg-muted px-1">{"{svg}"}</code> and{" "}
          <code className="rounded bg-muted px-1">{"{enrichment}"}</code> are
          replaced with actual data at runtime.
        </p>
        <pre className="overflow-auto whitespace-pre-wrap rounded border border-border bg-card p-4 font-mono text-[11px] leading-relaxed text-muted-foreground">
          {prompts[activeKey] ?? "No prompt found"}
        </pre>
      </div>
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

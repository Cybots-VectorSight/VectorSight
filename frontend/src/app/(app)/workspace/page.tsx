"use client"

import { useState, useCallback, useRef, useEffect } from "react"
import { SvgPreview } from "@/components/svg-preview"
import { SvgInput } from "@/components/svg-input"
import { ChatPanel } from "@/components/chat-panel"
import { PipelinePanel } from "@/components/pipeline-panel"
import { ProcessingDialog } from "@/components/processing-dialog"
import { VersionTimeline } from "@/components/version-timeline"
import { BaselineToggle } from "@/components/baseline-toggle"
import { useWorkspace } from "@/hooks/use-workspace"
import { parseSSE } from "@/lib/sse"
import type { TransformProgress, StreamingAnalyzeResult } from "@/lib/types"

const MIN_WIDTH = 320
const MAX_WIDTH = 640
const DEFAULT_WIDTH = 420

type Tab = "chat" | "pipeline"
type ProcessingPhase = "idle" | "streaming" | "complete" | "error"

export default function WorkspacePage() {
  const {
    state,
    loadSvg,
    addVersion,
    setActiveVersion,
    addMessage,
    updateMessage,
    setBaseline,
    setLoading,
  } = useWorkspace()

  const [panelWidth, setPanelWidth] = useState(DEFAULT_WIDTH)
  const [activeTab, setActiveTab] = useState<Tab>("chat")
  const isDragging = useRef(false)

  // Pipeline streaming state (lifted from PipelinePanel)
  const [processingPhase, setProcessingPhase] = useState<ProcessingPhase>("idle")
  const [pipelineTransforms, setPipelineTransforms] = useState<TransformProgress[]>([])
  const [pipelineResult, setPipelineResult] = useState<StreamingAnalyzeResult | null>(null)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const analyzedSvgRef = useRef<string | null>(null)

  // Cache: SVG content → enrichment result (avoids re-running pipeline on version switch)
  const enrichmentCacheRef = useRef<Map<string, { result: StreamingAnalyzeResult; transforms: TransformProgress[] }>>(new Map())

  // Run streaming analysis when SVG changes
  const runAnalysis = useCallback(async (svgContent: string) => {
    // Check cache first
    const cached = enrichmentCacheRef.current.get(svgContent)
    if (cached) {
      setPipelineResult(cached.result)
      setPipelineTransforms(cached.transforms)
      setPipelineError(null)
      analyzedSvgRef.current = svgContent
      setProcessingPhase("idle")
      return
    }

    setProcessingPhase("streaming")
    setPipelineError(null)
    setPipelineResult(null)
    setPipelineTransforms([])

    const collectedTransforms: TransformProgress[] = []

    try {
      const res = await fetch("/api/analyze/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ svg: svgContent }),
      })

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ error: "Request failed" }))
        throw new Error(err.error || err.message || `API error ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let sseEvent = ""
      let finalResult: StreamingAnalyzeResult | null = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const { events, remaining, currentEvent } = parseSSE(buffer, sseEvent)
        buffer = remaining
        sseEvent = currentEvent

        for (const { event, data } of events) {
          try {
            const parsed = JSON.parse(data)
            if (event === "progress") {
              const tp = parsed as TransformProgress
              const idx = collectedTransforms.findIndex((t) => t.transform_id === tp.transform_id)
              if (idx !== -1) {
                collectedTransforms[idx] = tp
              } else {
                collectedTransforms.push(tp)
              }
              setPipelineTransforms([...collectedTransforms])
            } else if (event === "result") {
              finalResult = parsed as StreamingAnalyzeResult
              setPipelineResult(finalResult)
            } else if (event === "error") {
              throw new Error(parsed.message || parsed.content || "Pipeline error")
            }
          } catch (e) {
            if (e instanceof SyntaxError) continue
            throw e
          }
        }
      }

      analyzedSvgRef.current = svgContent
      // Cache the result
      if (finalResult) {
        enrichmentCacheRef.current.set(svgContent, { result: finalResult, transforms: collectedTransforms })
      }
      setProcessingPhase("complete")
    } catch (err) {
      setPipelineError(err instanceof Error ? err.message : "Something went wrong")
      setProcessingPhase("error")
    }
  }, [])

  // Trigger analysis when SVG loads
  useEffect(() => {
    if (state.currentSvg && state.currentSvg !== analyzedSvgRef.current) {
      runAnalysis(state.currentSvg)
    }
  }, [state.currentSvg, runAnalysis])

  const handleDialogClose = useCallback(() => {
    setProcessingPhase("idle")
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isDragging.current = true

    const onMouseMove = (ev: MouseEvent) => {
      if (!isDragging.current) return
      const newWidth = ev.clientX - 240
      setPanelWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, newWidth)))
    }

    const onMouseUp = () => {
      isDragging.current = false
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }

    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
  }, [])

  return (
    <div className="flex h-full flex-col">
      {/* Processing dialog — blocks interaction until done */}
      {(processingPhase === "streaming" || processingPhase === "complete" || processingPhase === "error") && (
        <ProcessingDialog
          phase={processingPhase}
          transforms={pipelineTransforms}
          result={pipelineResult}
          error={pipelineError}
          onClose={handleDialogClose}
        />
      )}

      {/* Toolbar */}
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-border px-4">
        <h1 className="text-sm font-medium">Workspace</h1>
        <BaselineToggle
          enabled={state.baselineEnabled}
          onToggle={setBaseline}
        />
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Left panel — SVG Preview + Input + Timeline */}
        <div
          className="flex shrink-0 flex-col"
          style={{ width: panelWidth }}
        >
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <SvgInput onLoad={loadSvg} hasSvg={!!state.currentSvg} currentSvg={state.currentSvg ?? undefined} />
            <SvgPreview
              svg={state.currentSvg}
              versions={state.versions}
              activeVersionId={state.activeVersionId}
              onSetActiveVersion={setActiveVersion}
            />
            <VersionTimeline
              versions={state.versions}
              activeVersionId={state.activeVersionId}
              onSelectVersion={setActiveVersion}
            />
          </div>
        </div>

        {/* Resize handle */}
        <div
          onMouseDown={handleMouseDown}
          className="group relative flex w-0 shrink-0 cursor-col-resize items-stretch"
        >
          <div className="absolute inset-y-0 left-0 w-px bg-border transition-colors group-hover:bg-foreground/30" />
          <div className="absolute inset-y-0 -left-1.5 w-3" />
        </div>

        {/* Right panel — Chat / Pipeline tabs */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Tab bar */}
          <div className="flex h-9 shrink-0 items-stretch border-b border-border">
            <button
              onClick={() => setActiveTab("chat")}
              className={`px-4 text-xs font-medium transition-colors ${
                activeTab === "chat"
                  ? "border-b-2 border-foreground -mb-px text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Chat
            </button>
            <button
              onClick={() => setActiveTab("pipeline")}
              disabled={!pipelineResult}
              className={`px-4 text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                activeTab === "pipeline"
                  ? "border-b-2 border-foreground -mb-px text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Pipeline
            </button>
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === "chat" ? (
              <ChatPanel
                svg={state.currentSvg}
                messages={state.messages}
                isLoading={state.isLoading}
                baselineEnabled={state.baselineEnabled}
                versions={state.versions}
                enrichment={pipelineResult?.enrichment?.enrichment_text ?? null}
                onAddMessage={addMessage}
                onUpdateMessage={updateMessage}
                onSetLoading={setLoading}
                onAddVersion={addVersion}
                onLoadSvg={loadSvg}
              />
            ) : (
              <PipelinePanel
                data={pipelineResult}
                transforms={pipelineTransforms}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

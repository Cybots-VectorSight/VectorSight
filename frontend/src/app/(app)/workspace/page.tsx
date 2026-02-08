"use client"

import { useState, useCallback, useRef } from "react"
import { SvgPreview } from "@/components/svg-preview"
import { SvgInput } from "@/components/svg-input"
import { ChatPanel } from "@/components/chat-panel"
import { VersionTimeline } from "@/components/version-timeline"
import { BaselineToggle } from "@/components/baseline-toggle"
import { useWorkspace } from "@/hooks/use-workspace"

const MIN_WIDTH = 320
const MAX_WIDTH = 640
const DEFAULT_WIDTH = 420

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
  const isDragging = useRef(false)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isDragging.current = true

    const onMouseMove = (ev: MouseEvent) => {
      if (!isDragging.current) return
      // Subtract sidebar width (240px = w-60)
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
          {/* Visual border line */}
          <div className="absolute inset-y-0 left-0 w-px bg-border transition-colors group-hover:bg-foreground/30" />
          {/* Wider hit area */}
          <div className="absolute inset-y-0 -left-1.5 w-3" />
        </div>

        {/* Right panel — Chat */}
        <div className="flex flex-1 flex-col">
          <ChatPanel
            svg={state.currentSvg}
            messages={state.messages}
            isLoading={state.isLoading}
            baselineEnabled={state.baselineEnabled}
            versions={state.versions}
            onAddMessage={addMessage}
            onUpdateMessage={updateMessage}
            onSetLoading={setLoading}
            onAddVersion={addVersion}
            onLoadSvg={loadSvg}
          />
        </div>
      </div>
    </div>
  )
}

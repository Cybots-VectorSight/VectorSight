"use client"

import { useMemo, useState, useRef, useEffect } from "react"
import {
  Grid3X3,
  Maximize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  ChevronDown,
  Check,
} from "lucide-react"
import {
  TransformWrapper,
  TransformComponent,
  useControls,
} from "react-zoom-pan-pinch"
import { sanitizeSvg } from "@/lib/sanitize"
import type { SvgVersion } from "@/lib/types"

interface SvgPreviewProps {
  svg: string | null
  versions?: SvgVersion[]
  activeVersionId?: string | null
  onSetActiveVersion?: (id: string) => void
  className?: string
}

function ZoomControls() {
  const { zoomIn, zoomOut, resetTransform } = useControls()
  return (
    <div className="flex gap-0.5">
      <button
        onClick={() => zoomIn()}
        className="rounded-md p-1.5 bg-background/60 backdrop-blur-sm text-muted-foreground hover:text-foreground transition-colors"
        title="Zoom in"
      >
        <ZoomIn className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => zoomOut()}
        className="rounded-md p-1.5 bg-background/60 backdrop-blur-sm text-muted-foreground hover:text-foreground transition-colors"
        title="Zoom out"
      >
        <ZoomOut className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => resetTransform()}
        className="rounded-md p-1.5 bg-background/60 backdrop-blur-sm text-muted-foreground hover:text-foreground transition-colors"
        title="Reset view"
      >
        <RotateCcw className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

export function SvgPreview({
  svg,
  versions,
  activeVersionId,
  onSetActiveVersion,
  className = "",
}: SvgPreviewProps) {
  const sanitized = useMemo(() => (svg ? sanitizeSvg(svg) : null), [svg])
  const [showGrid, setShowGrid] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const activeIndex = versions?.findIndex((v) => v.id === activeVersionId) ?? -1
  const hasVersions = versions && versions.length > 1

  // Close dropdown on outside click
  useEffect(() => {
    if (!dropdownOpen) return
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [dropdownOpen])

  if (!sanitized) {
    return (
      <div
        className={`flex aspect-square items-center justify-center rounded-lg border border-dashed border-border text-muted-foreground/50 ${className}`}
      >
        <div className="flex flex-col items-center gap-2">
          <Maximize2 className="h-6 w-6" />
          <p className="text-xs">No SVG loaded</p>
        </div>
      </div>
    )
  }

  return (
    <div
      ref={wrapperRef}
      className={`relative rounded-lg border border-border overflow-hidden ${className}`}
    >
      <TransformWrapper
        initialScale={1}
        minScale={0.25}
        maxScale={8}
        centerOnInit
        wheel={{ step: 0.1 }}
      >
        <div className="absolute right-2 top-2 z-10 flex gap-1">
          <ZoomControls />
          <button
            onClick={() => setShowGrid((v) => !v)}
            className={`rounded-md p-1.5 backdrop-blur-sm transition-colors ${
              showGrid
                ? "bg-foreground/20 text-foreground"
                : "bg-background/60 text-muted-foreground hover:text-foreground"
            }`}
            title="Toggle grid background"
          >
            <Grid3X3 className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* SVG render area */}
        <div
          className={`aspect-square ${
            showGrid
              ? "bg-[length:20px_20px] bg-[image:linear-gradient(to_right,var(--border)_1px,transparent_1px),linear-gradient(to_bottom,var(--border)_1px,transparent_1px)]"
              : "bg-card"
          }`}
        >
          <TransformComponent
            wrapperStyle={{ width: "100%", height: "100%" }}
            contentStyle={{
              width: "100%",
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <div
              className="p-6 [&>svg]:max-h-full [&>svg]:max-w-full"
              dangerouslySetInnerHTML={{ __html: sanitized }}
            />
          </TransformComponent>
        </div>
      </TransformWrapper>

      {/* Version dropdown */}
      {hasVersions && (
        <div ref={dropdownRef} className="relative border-t border-border">
          <button
            onClick={() => setDropdownOpen((v) => !v)}
            className="flex w-full items-center justify-center gap-1.5 bg-card/80 backdrop-blur-sm px-3 py-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
          >
            <span>
              {activeIndex === 0 ? "Original" : `v${activeIndex + 1}`}
              {versions[activeIndex]?.label && versions[activeIndex].label !== "Original"
                ? ` — ${versions[activeIndex].label}`
                : ""}
            </span>
            <ChevronDown
              className={`h-3 w-3 transition-transform ${dropdownOpen ? "rotate-180" : ""}`}
            />
          </button>

          {dropdownOpen && (
            <div className="absolute bottom-full left-0 right-0 z-20 border border-border rounded-t-md bg-card shadow-lg overflow-hidden">
              {versions.map((v, i) => {
                const label = i === 0 ? "Original" : `v${i + 1}`
                const isActive = v.id === activeVersionId
                return (
                  <button
                    key={v.id}
                    onClick={() => {
                      onSetActiveVersion?.(v.id)
                      setDropdownOpen(false)
                    }}
                    className={`flex w-full items-center justify-between px-3 py-1.5 text-xs transition-colors ${
                      isActive
                        ? "bg-accent text-foreground"
                        : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                    }`}
                  >
                    <span className="truncate">
                      {label}
                      {v.label && v.label !== "Original" ? ` — ${v.label}` : ""}
                    </span>
                    {isActive && <Check className="h-3 w-3 shrink-0 ml-2" />}
                  </button>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

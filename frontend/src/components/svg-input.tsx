"use client"

import { useState, useCallback, useMemo } from "react"
import {
  Upload,
  Code,
  Grid3X3,
  ChevronDown,
  ChevronUp,
  Clipboard,
  FileUp,
  X,
  EyeOff,
  Eye,
} from "lucide-react"
import { useDropzone } from "react-dropzone"
import Editor from "react-simple-code-editor"
import { Button } from "@/components/ui/button"
import { sampleSvgs } from "@/lib/samples"
import { sanitizeSvg } from "@/lib/sanitize"
import { anonymizeSvg } from "@/lib/anonymize"

type Tab = "drop" | "paste" | "samples"

interface SvgInputProps {
  onLoad: (svg: string) => void
  hasSvg: boolean
  currentSvg?: string
}

function highlightSvg(code: string): string {
  const escaped = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")

  // Single-pass to avoid span tags being re-processed by later regexes
  return escaped.replace(
    /(&lt;\/?)([\w-]+)|([\w-]+)(=)|(".*?")/g,
    (match, tagOpen, tagName, attrName, eq, strVal) => {
      if (tagName)
        return `${tagOpen}<span style="color:var(--foreground)">${tagName}</span>`
      if (attrName)
        return `<span style="color:var(--muted-foreground)">${attrName}</span>${eq}`
      if (strVal) return `<span style="opacity:0.6">${strVal}</span>`
      return match
    }
  )
}

export function SvgInput({ onLoad, hasSvg, currentSvg }: SvgInputProps) {
  const [activeTab, setActiveTab] = useState<Tab>("drop")
  const [pasteValue, setPasteValue] = useState("")
  const [collapsed, setCollapsed] = useState(hasSvg)
  const [dropError, setDropError] = useState<string | null>(null)
  const [showAnonymized, setShowAnonymized] = useState(false)

  const anonymized = useMemo(() => {
    const source = currentSvg || pasteValue
    if (!source.trim()) return null
    return anonymizeSvg(source)
  }, [currentSvg, pasteValue])

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      setDropError(null)
      const file = acceptedFiles[0]
      if (!file) return
      const reader = new FileReader()
      reader.onload = (ev) => {
        const text = ev.target?.result as string
        if (text) {
          onLoad(text)
          setCollapsed(true)
        }
      }
      reader.readAsText(file)
    },
    [onLoad]
  )

  const onDropRejected = useCallback(() => {
    setDropError("Only .svg files are supported")
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: { "image/svg+xml": [".svg"] },
    maxFiles: 1,
    multiple: false,
  })

  const handlePasteFromClipboard = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText()
      if (text.trim().includes("<svg")) {
        onLoad(text.trim())
        setCollapsed(true)
      } else {
        setPasteValue(text)
        setActiveTab("paste")
      }
    } catch {
      setActiveTab("paste")
    }
  }, [onLoad])

  const handlePasteSubmit = useCallback(() => {
    if (pasteValue.trim()) {
      onLoad(pasteValue.trim())
      setCollapsed(true)
    }
  }, [pasteValue, onLoad])

  if (collapsed && hasSvg) {
    return (
      <Button
        variant="outline"
        size="sm"
        onClick={() => setCollapsed(false)}
        className="w-full"
      >
        <ChevronDown className="mr-2 h-3 w-3" />
        Change SVG
      </Button>
    )
  }

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    {
      id: "drop",
      label: "Drop / Browse",
      icon: <Upload className="h-3 w-3" />,
    },
    { id: "paste", label: "Code", icon: <Code className="h-3 w-3" /> },
    {
      id: "samples",
      label: "Samples",
      icon: <Grid3X3 className="h-3 w-3" />,
    },
  ]

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex gap-0.5 rounded-md bg-muted p-0.5">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id)
                setDropError(null)
              }}
              className={`flex items-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors ${
                activeTab === tab.id
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
        {hasSvg && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setCollapsed(true)}
            className="h-7 px-2"
          >
            <ChevronUp className="h-3 w-3" />
          </Button>
        )}
      </div>

      {/* Drop / Browse — react-dropzone */}
      {activeTab === "drop" && (
        <div>
          <div
            {...getRootProps()}
            className={`relative flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-all ${
              isDragActive
                ? "border-foreground bg-accent/50 scale-[1.02]"
                : "border-border hover:border-muted-foreground"
            }`}
          >
            <input {...getInputProps()} />
            <div
              className={`mb-3 flex h-12 w-12 items-center justify-center rounded-xl border border-border transition-colors ${
                isDragActive ? "bg-foreground/10" : "bg-muted"
              }`}
            >
              <FileUp
                className={`h-5 w-5 transition-colors ${
                  isDragActive ? "text-foreground" : "text-muted-foreground"
                }`}
              />
            </div>
            <p className="text-sm font-medium">
              {isDragActive ? "Drop your SVG here" : "Drag & drop SVG file"}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              or click to browse
            </p>
          </div>
          <div className="mt-3 flex gap-2">
            <Button
              variant="outline"
              size="sm"
              className="flex-1"
              onClick={handlePasteFromClipboard}
            >
              <Clipboard className="mr-1.5 h-3 w-3" />
              Paste from Clipboard
            </Button>
          </div>
          {dropError && (
            <div className="mt-3 flex items-center gap-1.5 text-xs text-destructive">
              <X className="h-3 w-3" />
              {dropError}
            </div>
          )}
        </div>
      )}

      {/* Code editor — react-simple-code-editor */}
      {activeTab === "paste" && (
        <div className="space-y-2">
          {/* Anonymized toggle — only when there's SVG content */}
          {(currentSvg || pasteValue.trim()) && anonymized && (
            <div className="flex items-center justify-between">
              <button
                onClick={() => setShowAnonymized(!showAnonymized)}
                className={`flex items-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors ${
                  showAnonymized
                    ? "bg-foreground text-background"
                    : "bg-muted text-muted-foreground hover:text-foreground"
                }`}
              >
                {showAnonymized ? (
                  <EyeOff className="h-3 w-3" />
                ) : (
                  <Eye className="h-3 w-3" />
                )}
                {showAnonymized ? "Anonymized" : "Show Anonymized"}
              </button>
              {showAnonymized && (
                <span className="text-[10px] text-muted-foreground">
                  {anonymized.summary}
                </span>
              )}
            </div>
          )}

          <div className="relative rounded-md border border-border bg-muted/30 overflow-hidden">
            {!showAnonymized && pasteValue && (
              <button
                onClick={() => setPasteValue("")}
                className="absolute right-2 top-2 z-10 rounded-sm p-0.5 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3 w-3" />
              </button>
            )}
            {showAnonymized && anonymized ? (
              <Editor
                value={anonymized.cleanSvg}
                onValueChange={() => {}}
                highlight={highlightSvg}
                padding={12}
                style={{
                  fontFamily: "var(--font-mono, ui-monospace, monospace)",
                  fontSize: 12,
                  lineHeight: 1.6,
                  minHeight: 140,
                  color: "var(--foreground)",
                  opacity: 0.8,
                }}
                textareaClassName="focus:outline-none"
                readOnly
              />
            ) : (
              <Editor
                value={pasteValue}
                onValueChange={setPasteValue}
                highlight={highlightSvg}
                placeholder={'<svg xmlns="http://www.w3.org/2000/svg" ...>\n  ...\n</svg>'}
                padding={12}
                style={{
                  fontFamily: "var(--font-mono, ui-monospace, monospace)",
                  fontSize: 12,
                  lineHeight: 1.6,
                  minHeight: 140,
                  color: "var(--foreground)",
                }}
                textareaClassName="focus:outline-none"
              />
            )}
          </div>
          {!showAnonymized && (
            <Button
              size="sm"
              onClick={handlePasteSubmit}
              disabled={!pasteValue.trim()}
              className="w-full"
            >
              Load SVG
            </Button>
          )}
        </div>
      )}

      {/* Samples */}
      {activeTab === "samples" && (
        <div className="grid grid-cols-3 gap-2">
          {sampleSvgs.map((sample) => (
            <button
              key={sample.id}
              onClick={() => {
                onLoad(sample.svg)
                setCollapsed(true)
              }}
              className="group flex flex-col items-center gap-1.5 rounded-lg border border-border p-3 transition-all hover:bg-accent hover:border-muted-foreground"
            >
              <div
                className="h-12 w-12 [&>svg]:h-full [&>svg]:w-full transition-transform group-hover:scale-110"
                dangerouslySetInnerHTML={{
                  __html: sanitizeSvg(sample.svg),
                }}
              />
              <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">
                {sample.name}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

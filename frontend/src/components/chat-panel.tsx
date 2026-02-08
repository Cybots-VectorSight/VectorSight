"use client"

import { useState, useCallback, useRef } from "react"
import { Send, Square, Sparkles, Paperclip } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputActions,
} from "@/components/prompt-kit/prompt-input"
import { Markdown } from "@/components/prompt-kit/markdown"
import { PromptSuggestion } from "@/components/prompt-kit/prompt-suggestion"
import {
  ChatContainerRoot,
  ChatContainerContent,
  ChatContainerScrollAnchor,
} from "@/components/prompt-kit/chat-container"
import { ScrollButton } from "@/components/prompt-kit/scroll-button"
import { ThinkingBar } from "@/components/prompt-kit/thinking-bar"
import {
  Reasoning,
  ReasoningTrigger,
  ReasoningContent,
} from "@/components/prompt-kit/reasoning"
import { Loader } from "@/components/prompt-kit/loader"
import {
  FileUpload,
  FileUploadTrigger,
  FileUploadContent,
} from "@/components/prompt-kit/file-upload"
import { SvgArtifact } from "@/components/svg-artifact"
import type { ChatMessage, SvgVersion } from "@/lib/types"
import { detectIntent } from "@/lib/intent"
import { generateId } from "@/lib/utils"

const SUGGESTIONS = [
  "What shapes are in this SVG?",
  "How many elements does it have?",
  "Make it larger and add color",
  "Describe the spatial layout",
]

interface ChatPanelProps {
  svg: string | null
  messages: ChatMessage[]
  isLoading: boolean
  baselineEnabled: boolean
  versions: SvgVersion[]
  onAddMessage: (message: ChatMessage) => void
  onUpdateMessage: (id: string, updates: Partial<ChatMessage>) => void
  onSetLoading: (loading: boolean) => void
  onAddVersion: (version: SvgVersion) => void
  onLoadSvg?: (svg: string) => void
}

/** Parse SSE text into events */
function parseSSE(buffer: string): { events: Array<{ event: string; data: string }>; remaining: string } {
  const events: Array<{ event: string; data: string }> = []
  const lines = buffer.split("\n")
  let currentEvent = ""
  let remaining = ""

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // If it's the last line and doesn't end with \n, it's incomplete
    if (i === lines.length - 1 && !buffer.endsWith("\n")) {
      remaining = line
      break
    }

    if (line.startsWith("event: ")) {
      currentEvent = line.slice(7).trim()
    } else if (line.startsWith("data: ")) {
      const data = line.slice(6)
      events.push({ event: currentEvent || "message", data })
      currentEvent = ""
    } else if (line === "") {
      // Empty line = end of event (already pushed)
    }
  }

  return { events, remaining }
}

export function ChatPanel({
  svg,
  messages,
  isLoading,
  baselineEnabled,
  versions,
  onAddMessage,
  onUpdateMessage,
  onSetLoading,
  onAddVersion,
  onLoadSvg,
}: ChatPanelProps) {
  const [input, setInput] = useState("")
  const abortControllerRef = useRef<AbortController | null>(null)

  const handleFilesAdded = useCallback((files: File[]) => {
    const file = files[0]
    if (!file) return
    if (file.type === "image/svg+xml" || file.name.endsWith(".svg")) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        if (text && onLoadSvg) onLoadSvg(text)
      }
      reader.readAsText(file)
    }
  }, [onLoadSvg])

  const handleStop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    onSetLoading(false)
  }, [onSetLoading])

  const handleSubmit = useCallback(async () => {
    if (!input.trim() || !svg || isLoading) return

    const { intent, cleanInput } = detectIntent(input)
    const userMessage: ChatMessage = {
      id: generateId(),
      role: "user",
      content: input,
      intent,
      timestamp: Date.now(),
    }
    onAddMessage(userMessage)
    setInput("")
    onSetLoading(true)

    const assistantId = generateId()
    const placeholderMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      intent,
      timestamp: Date.now(),
    }
    onAddMessage(placeholderMessage)

    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      if (intent === "chat") {
        // Use streaming endpoint
        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ svg, question: cleanInput }),
          signal: controller.signal,
        })

        if (!res.ok || !res.body) {
          const err = await res.json().catch(() => ({ error: "Request failed" }))
          throw new Error(err.error || err.message || `API error ${res.status}`)
        }

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        let thinkingAccum = ""
        let responseAccum = ""

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const { events, remaining } = parseSSE(buffer)
          buffer = remaining

          for (const { event, data } of events) {
            try {
              const parsed = JSON.parse(data)
              if (event === "thinking") {
                thinkingAccum += parsed.content
                onUpdateMessage(assistantId, {
                  reasoning: thinkingAccum,
                })
              } else if (event === "response") {
                responseAccum += parsed.content
                onUpdateMessage(assistantId, {
                  content: responseAccum,
                  reasoning: thinkingAccum || undefined,
                })
              } else if (event === "error") {
                throw new Error(parsed.content)
              }
              // "done" event = stream complete
            } catch (e) {
              if (e instanceof SyntaxError) continue // skip malformed JSON
              throw e
            }
          }
        }

        // Final update to ensure everything is set
        if (responseAccum) {
          onUpdateMessage(assistantId, {
            content: responseAccum,
            reasoning: thinkingAccum || undefined,
          })
        }
      } else {
        // Modify intent — non-streaming
        const res = await fetch("/api/modify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ svg, instruction: cleanInput }),
          signal: controller.signal,
        })
        if (!res.ok) {
          const err = await res.json().catch(() => ({ error: "Request failed" }))
          throw new Error(err.error || err.message || `API error ${res.status}`)
        }
        const result = await res.json()

        const version: SvgVersion = {
          id: generateId(),
          svg: result.svg,
          label: cleanInput.slice(0, 30),
          timestamp: Date.now(),
        }
        onAddVersion(version)

        const changesText = result.changes?.length
          ? "\n\n**Changes:**\n" + result.changes.map((c: string) => `- ${c}`).join("\n")
          : ""

        onUpdateMessage(assistantId, {
          content: `SVG modified successfully.${changesText}`,
          svgVersionId: version.id,
          svgChanges: result.changes || [],
        })
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        onUpdateMessage(assistantId, {
          content: "Stopped by user.",
        })
      } else {
        onUpdateMessage(assistantId, {
          content: `Error: ${err instanceof Error ? err.message : "Something went wrong"}`,
        })
      }
    } finally {
      abortControllerRef.current = null
      onSetLoading(false)
    }
  }, [input, svg, isLoading, onAddMessage, onUpdateMessage, onSetLoading, onAddVersion])

  if (!svg) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <p className="text-sm text-muted-foreground">
          Load an SVG to start chatting
        </p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <ChatContainerRoot className="relative flex-1">
        <ChatContainerContent className="gap-4 px-4 py-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-4 pt-12">
              <Sparkles className="h-8 w-8 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground text-center max-w-xs">
                Ask questions about your SVG or request modifications
              </p>
              <div className="grid w-full gap-2 max-w-sm">
                {SUGGESTIONS.map((s) => (
                  <PromptSuggestion
                    key={s}
                    onClick={() => setInput(s)}
                  >
                    {s}
                  </PromptSuggestion>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => {
            if (msg.role === "user") {
              return (
                <div key={msg.id} className="flex justify-end">
                  <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary text-primary-foreground px-4 py-2 text-sm">
                    {msg.content}
                  </div>
                </div>
              )
            }

            // Assistant message
            const isThinking = !msg.content && isLoading
            const isStreaming = isLoading && msg === messages[messages.length - 1]
            const hasThinking = !!msg.reasoning

            return (
              <div key={msg.id} className="space-y-2">
                {/* Active thinking indicator (no content yet) */}
                {isThinking && !hasThinking && (
                  <ThinkingBar
                    text="Analyzing SVG"
                    onStop={handleStop}
                    stopLabel="Stop"
                  />
                )}

                {/* Chain of thought — persists after completion */}
                {hasThinking && (
                  <Reasoning isStreaming={isStreaming && !msg.content}>
                    <div className="flex items-center gap-2">
                      {isStreaming && !msg.content && (
                        <Loader variant="circular" size="sm" />
                      )}
                      <ReasoningTrigger className="text-xs text-muted-foreground hover:text-foreground">
                        {isStreaming && !msg.content ? "Thinking..." : "View thinking"}
                      </ReasoningTrigger>
                    </div>
                    <ReasoningContent
                      className="mt-1"
                      contentClassName="text-xs text-muted-foreground/80"
                      markdown
                    >
                      {msg.reasoning!}
                    </ReasoningContent>
                  </Reasoning>
                )}

                {/* AI answer — on canvas, no bubble */}
                {msg.content && (
                  <div className="max-w-[95%] text-sm">
                    <Markdown className="prose prose-sm dark:prose-invert prose-p:leading-relaxed prose-pre:bg-card">
                      {msg.content}
                    </Markdown>
                  </div>
                )}

                {/* Baseline comparison */}
                {msg.baselineContent && (
                  <div className="mt-3 space-y-2">
                    <div className="rounded-lg border border-border bg-card p-3">
                      <p className="mb-1 text-xs font-medium text-muted-foreground">
                        With VectorSight
                      </p>
                      <div className="text-sm">
                        <Markdown>{msg.content}</Markdown>
                      </div>
                    </div>
                    <div className="rounded-lg border border-dashed border-border bg-card/50 p-3">
                      <p className="mb-1 text-xs font-medium text-muted-foreground">
                        Without VectorSight
                      </p>
                      <div className="text-sm">
                        <Markdown>{msg.baselineContent}</Markdown>
                      </div>
                    </div>
                  </div>
                )}

                {/* SVG artifact indicator */}
                {msg.svgVersionId && (() => {
                  const artifactVersion = versions.find((v) => v.id === msg.svgVersionId)
                  if (!artifactVersion) return null
                  return (
                    <SvgArtifact
                      version={artifactVersion}
                      versions={versions}
                      changes={msg.svgChanges}
                    />
                  )
                })()}
              </div>
            )
          })}

          <ChatContainerScrollAnchor />
        </ChatContainerContent>

        {/* Scroll to bottom */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center pb-2">
          <div className="pointer-events-auto">
            <ScrollButton />
          </div>
        </div>
      </ChatContainerRoot>

      {/* Input area */}
      <div className="border-t border-border p-4">
        <FileUpload onFilesAdded={handleFilesAdded} accept=".svg,image/svg+xml" multiple={false}>
          <PromptInput
            value={input}
            onValueChange={setInput}
            onSubmit={handleSubmit}
            isLoading={isLoading}
            disabled={isLoading}
          >
            <PromptInputTextarea
              placeholder="Ask about your SVG or describe changes..."
              className="border-0 dark:bg-transparent min-h-0 shadow-none focus-visible:ring-0 focus-visible:border-transparent"
            />
            <PromptInputActions className="justify-between px-2 pb-1">
              <FileUploadTrigger
                className="rounded-md p-1.5 text-muted-foreground hover:text-foreground transition-colors"
                title="Attach SVG file"
              >
                <Paperclip className="h-4 w-4" />
              </FileUploadTrigger>
              {isLoading ? (
                <Button
                  size="icon"
                  variant="destructive"
                  className="h-8 w-8 rounded-full"
                  onClick={handleStop}
                >
                  <Square className="h-3.5 w-3.5 fill-current" />
                </Button>
              ) : (
                <Button
                  size="icon"
                  className="h-8 w-8 rounded-full"
                  onClick={handleSubmit}
                  disabled={!input.trim()}
                >
                  <Send className="h-4 w-4" />
                </Button>
              )}
            </PromptInputActions>
          </PromptInput>
          <FileUploadContent>
            <div className="flex flex-col items-center gap-3 rounded-xl border-2 border-dashed border-border bg-card p-8">
              <Paperclip className="h-8 w-8 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Drop SVG file here</p>
            </div>
          </FileUploadContent>
        </FileUpload>
      </div>
    </div>
  )
}

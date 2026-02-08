"use client"

import { useRouter } from "next/navigation"
import { useWorkspaceHistory } from "@/hooks/use-workspace-history"
import { sanitizeSvg } from "@/lib/sanitize"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Trash2, MessageSquare, Clock, History } from "lucide-react"

export default function HistoryPage() {
  const { history, removeEntry, clearHistory } = useWorkspaceHistory()
  const router = useRouter()

  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-10 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">History</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Your recent workspace sessions.
          </p>
        </div>
        {history.length > 0 && (
          <Button variant="outline" size="sm" onClick={clearHistory}>
            <Trash2 className="mr-1.5 h-3 w-3" />
            Clear All
          </Button>
        )}
      </div>

      <Separator />

      {history.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <History className="mb-3 h-10 w-10 text-muted-foreground/30" />
          <p className="text-sm text-muted-foreground">No sessions yet</p>
          <p className="mt-1 text-xs text-muted-foreground/60">
            Your workspace sessions will appear here.
          </p>
          <Button
            variant="outline"
            size="sm"
            className="mt-4"
            onClick={() => router.push("/workspace")}
          >
            Open Workspace
          </Button>
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {history.map((entry) => (
            <button
              key={entry.id}
              onClick={() => router.push("/workspace")}
              className="group relative flex items-start gap-4 rounded-lg border border-border p-4 text-left transition-all hover:bg-accent/50 hover:border-muted-foreground"
            >
              {/* SVG thumbnail */}
              <div
                className="h-14 w-14 shrink-0 rounded-md border border-border bg-card p-1.5 [&>svg]:h-full [&>svg]:w-full transition-transform group-hover:scale-105"
                dangerouslySetInnerHTML={{
                  __html: sanitizeSvg(entry.svgSnippet),
                }}
              />

              {/* Info */}
              <div className="min-w-0 flex-1 space-y-1">
                <p className="text-sm font-medium truncate">{entry.label}</p>
                <p className="text-xs text-muted-foreground truncate">
                  {entry.query}
                </p>
                <div className="flex items-center gap-3 text-[10px] text-muted-foreground/60">
                  <span className="flex items-center gap-1">
                    <MessageSquare className="h-2.5 w-2.5" />
                    {entry.messageCount} message{entry.messageCount !== 1 ? "s" : ""}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-2.5 w-2.5" />
                    {new Date(entry.timestamp).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>
              </div>

              {/* Delete */}
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  removeEntry(entry.id)
                }}
                className="absolute right-2 top-2 rounded-sm p-1 text-muted-foreground/0 transition-colors group-hover:text-muted-foreground/50 hover:!text-foreground"
                title="Remove"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

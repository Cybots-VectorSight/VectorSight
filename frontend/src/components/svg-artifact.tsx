"use client"

import { Layers } from "lucide-react"
import type { SvgVersion } from "@/lib/types"

interface SvgArtifactProps {
  /** The version this artifact represents */
  version: SvgVersion
  /** All versions to compute version number */
  versions: SvgVersion[]
  /** List of changes made */
  changes?: string[]
}

export function SvgArtifact({
  version,
  versions,
  changes,
}: SvgArtifactProps) {
  const versionIndex = versions.findIndex((v) => v.id === version.id)
  const versionLabel = versionIndex === 0 ? "Original" : `v${versionIndex + 1}`

  return (
    <div className="mt-2 flex items-start gap-2 rounded-lg border border-border bg-card px-3 py-2">
      <Layers className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
      <div className="min-w-0 space-y-0.5">
        <p className="text-xs font-medium">
          {version.label || "SVG modified"}
          <span className="ml-1.5 text-[10px] text-muted-foreground">
            {versionLabel}
          </span>
        </p>
        {changes && changes.length > 0 && (
          <ul className="space-y-0">
            {changes.map((c, i) => (
              <li
                key={i}
                className="flex items-start gap-1.5 text-xs text-muted-foreground"
              >
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-foreground/30" />
                {c}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

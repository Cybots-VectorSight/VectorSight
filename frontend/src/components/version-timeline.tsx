"use client"

import type { SvgVersion } from "@/lib/types"
import { VersionThumbnail } from "@/components/version-thumbnail"

interface VersionTimelineProps {
  versions: SvgVersion[]
  activeVersionId: string | null
  onSelectVersion: (id: string) => void
}

export function VersionTimeline({
  versions,
  activeVersionId,
  onSelectVersion,
}: VersionTimelineProps) {
  if (versions.length <= 1) return null

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-medium text-muted-foreground">Versions</h3>
      <div className="flex gap-2 overflow-x-auto pb-2">
        {versions.map((v) => (
          <VersionThumbnail
            key={v.id}
            version={v}
            isActive={v.id === activeVersionId}
            onClick={() => onSelectVersion(v.id)}
          />
        ))}
      </div>
    </div>
  )
}

"use client"

import { cn } from "@/lib/utils"
import { sanitizeSvg } from "@/lib/sanitize"
import type { SvgVersion } from "@/lib/types"

interface VersionThumbnailProps {
  version: SvgVersion
  isActive: boolean
  onClick: () => void
}

export function VersionThumbnail({
  version,
  isActive,
  onClick,
}: VersionThumbnailProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex shrink-0 flex-col items-center gap-1 rounded-lg border p-1.5 transition-all hover:bg-accent",
        isActive
          ? "border-primary ring-1 ring-primary"
          : "border-border"
      )}
    >
      <div
        className="h-16 w-16 overflow-hidden rounded [&>svg]:h-full [&>svg]:w-full"
        dangerouslySetInnerHTML={{ __html: sanitizeSvg(version.svg) }}
      />
      <span className="max-w-[64px] truncate text-[10px] text-muted-foreground">
        {version.label}
      </span>
    </button>
  )
}

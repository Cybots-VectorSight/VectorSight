"use client"

import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"

interface BaselineToggleProps {
  enabled: boolean
  onToggle: (enabled: boolean) => void
}

export function BaselineToggle({ enabled, onToggle }: BaselineToggleProps) {
  return (
    <div className="flex items-center gap-2">
      <Switch
        id="baseline"
        checked={enabled}
        onCheckedChange={onToggle}
      />
      <Label htmlFor="baseline" className="text-xs cursor-pointer">
        Baseline
      </Label>
    </div>
  )
}

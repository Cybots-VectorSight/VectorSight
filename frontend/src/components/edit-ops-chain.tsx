"use client"

import { Brain, Trash2, Pencil, Plus } from "lucide-react"
import {
  ChainOfThought,
  ChainOfThoughtStep,
  ChainOfThoughtTrigger,
  ChainOfThoughtContent,
} from "@/components/prompt-kit/chain-of-thought"
import type { EditOp } from "@/lib/types"

interface EditOpsChainProps {
  reasoning?: string
  ops: EditOp[]
}

function opIcon(action: EditOp["action"]) {
  switch (action) {
    case "delete":
      return <Trash2 className="size-4 text-red-500" />
    case "modify":
      return <Pencil className="size-4 text-blue-500" />
    case "add":
      return <Plus className="size-4 text-green-500" />
  }
}

function opLabel(op: EditOp) {
  switch (op.action) {
    case "delete":
      return `Delete ${op.target ?? "element"}`
    case "modify": {
      const keys = op.attributes ? Object.keys(op.attributes).join(", ") : ""
      return `Modify ${op.target ?? "element"}${keys ? ` (${keys})` : ""}`
    }
    case "add":
      return `Add element${op.position ? ` ${op.position}` : ""}`
  }
}

export function EditOpsChain({ reasoning, ops }: EditOpsChainProps) {
  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-muted-foreground">Surgical Edit</p>
      <ChainOfThought>
        {reasoning && (
          <ChainOfThoughtStep>
            <ChainOfThoughtTrigger
              leftIcon={<Brain className="size-4 text-purple-500" />}
            >
              Spatial reasoning
            </ChainOfThoughtTrigger>
            <ChainOfThoughtContent>
              <p className="text-xs text-muted-foreground whitespace-pre-wrap">
                {reasoning}
              </p>
            </ChainOfThoughtContent>
          </ChainOfThoughtStep>
        )}
        {ops.map((op, i) => (
          <ChainOfThoughtStep key={i}>
            <ChainOfThoughtTrigger leftIcon={opIcon(op.action)}>
              {opLabel(op)}
            </ChainOfThoughtTrigger>
            <ChainOfThoughtContent>
              <div className="space-y-1 text-xs text-muted-foreground">
                {op.action === "modify" && op.attributes && (
                  <div className="space-y-0.5">
                    {Object.entries(op.attributes).map(([k, v]) => (
                      <div key={k}>
                        <span className="font-mono text-foreground/70">{k}</span>
                        {": "}
                        <span className="font-mono">{v}</span>
                      </div>
                    ))}
                  </div>
                )}
                {op.action === "add" && op.svg_fragment && (
                  <pre className="overflow-x-auto rounded bg-card p-2 font-mono text-[11px]">
                    {op.svg_fragment}
                  </pre>
                )}
                {op.action === "delete" && op.target && (
                  <p>Remove element <span className="font-mono text-foreground/70">{op.target}</span></p>
                )}
              </div>
            </ChainOfThoughtContent>
          </ChainOfThoughtStep>
        ))}
      </ChainOfThought>
    </div>
  )
}

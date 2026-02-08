import Image from "next/image"
import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"

const notTable = [
  { label: "Fine-tuning", description: "No model weights are modified" },
  { label: "RAG", description: "No vector database or retrieval" },
  { label: "Tool / MCP", description: "No function calling or tool use" },
  { label: "Prompt injection", description: "No hidden instructions in prompts" },
]

const architecture = [
  { layer: "L0", name: "Primitives", count: 7, description: "Basic element extraction" },
  { layer: "L1", name: "Per-Element", count: 21, description: "Size, position, shape metrics" },
  { layer: "L2", name: "Canvas", count: 7, description: "Bounding box, density, coverage" },
  { layer: "L3", name: "Relational", count: 21, description: "Pairwise spatial relationships" },
  { layer: "L4", name: "Composition", count: 5, description: "Symmetry, clusters, hierarchy" },
]

export default function AboutPage() {
  return (
    <main className="mx-auto max-w-3xl px-4 py-16 space-y-16">
      {/* Heading */}
      <section className="text-center space-y-4">
        <Badge variant="secondary">About</Badge>
        <h1 className="text-4xl font-bold tracking-tight">VectorSight</h1>
        <p className="text-muted-foreground max-w-lg mx-auto">
          A geometry engine that transforms SVG code into structured spatial
          data for LLM comprehension.
        </p>
      </section>

      <Separator />

      {/* Team */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <Image
            src="/cybots.svg"
            alt="Cybots"
            width={28}
            height={28}
            className="dark:invert"
          />
          <h2 className="text-xl font-semibold">Team Cybots</h2>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Built for The Strange Data Project hackathon. We explore how
          transforming unconventional data formats — in this case, SVG
          vector graphics — can give language models entirely new
          capabilities.
        </p>
      </section>

      <Separator />

      {/* What This Is NOT */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">What This Is NOT</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {notTable.map((item) => (
            <div
              key={item.label}
              className="rounded-lg border border-border bg-card p-4"
            >
              <div className="font-medium text-sm">{item.label}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {item.description}
              </div>
            </div>
          ))}
        </div>
        <p className="text-sm text-muted-foreground">
          VectorSight is pure data transformation. SVG code goes in,
          structured spatial JSON comes out. Any LLM can read it.
        </p>
      </section>

      <Separator />

      {/* Architecture */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Transform Architecture</h2>
        <p className="text-sm text-muted-foreground">
          61 transforms organized in 5 layers, executed via topological
          dependency sort:
        </p>
        <div className="space-y-2">
          {architecture.map((layer) => (
            <div
              key={layer.layer}
              className="flex items-center gap-4 rounded-lg border border-border bg-card px-4 py-3"
            >
              <Badge variant="outline" className="font-mono shrink-0">
                {layer.layer}
              </Badge>
              <div className="flex-1">
                <div className="text-sm font-medium">{layer.name}</div>
                <div className="text-xs text-muted-foreground">
                  {layer.description}
                </div>
              </div>
              <span className="text-sm font-mono text-muted-foreground">
                {layer.count}
              </span>
            </div>
          ))}
        </div>
      </section>

      <Separator />

      {/* CTA */}
      <section className="text-center space-y-4">
        <h2 className="text-xl font-semibold">Try It Out</h2>
        <Link href="/signup" className={buttonVariants()}>
          Get Started
          <ArrowRight className="ml-2 h-4 w-4" />
        </Link>
      </section>

      <footer className="border-t border-border py-6 text-center text-xs text-muted-foreground">
        VectorSight by Cybots &middot; The Strange Data Project
      </footer>
    </main>
  )
}

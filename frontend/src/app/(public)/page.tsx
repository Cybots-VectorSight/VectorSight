import Link from "next/link"
import Image from "next/image"
import { ArrowRight, Code, Cpu, MessageSquare } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"

const steps = [
  {
    icon: Code,
    title: "SVG Code",
    description: "Paste or upload any SVG — icons, charts, diagrams, logos.",
  },
  {
    icon: Cpu,
    title: "Geometry Engine",
    description:
      "61 transforms across 5 layers extract spatial relationships, symmetry, topology.",
  },
  {
    icon: MessageSquare,
    title: "LLM Reads Structure",
    description:
      "Enriched spatial JSON gives any LLM deep geometric understanding.",
  },
]

const stats = [
  { value: "61", label: "Transforms" },
  { value: "5", label: "Layers" },
  { value: "~1,200", label: "Tokens" },
  { value: "Any", label: "LLM" },
]

export default function LandingPage() {
  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] flex-col">
      {/* Hero */}
      <section className="flex flex-1 flex-col items-center justify-center gap-8 px-4 py-24 text-center">
        <Badge variant="secondary" className="gap-1.5">
          The Strange Data Project
        </Badge>
        <div className="flex items-center gap-3">
          <Image
            src="/vectorsight-light.svg"
            alt="VectorSight"
            width={48}
            height={48}
            className="hidden dark:block"
          />
          <Image
            src="/vectorsight-dark.svg"
            alt="VectorSight"
            width={48}
            height={48}
            className="block dark:hidden"
          />
          <h1 className="text-5xl font-bold tracking-tight sm:text-6xl">
            VectorSight
          </h1>
        </div>
        <p className="max-w-xl text-lg text-muted-foreground">
          Transform SVG code into structured spatial data that LLMs can actually
          understand. No fine-tuning. No RAG. Pure data transformation.
        </p>
        <div className="flex gap-3">
          <Link
            href="/signup"
            className={buttonVariants({ size: "lg" })}
          >
            Get Started
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
          <Link
            href="/about"
            className={buttonVariants({ variant: "outline", size: "lg" })}
          >
            Learn More
          </Link>
        </div>
      </section>

      <Separator />

      {/* How It Works */}
      <section className="py-20 px-4">
        <div className="mx-auto max-w-4xl">
          <h2 className="mb-12 text-center text-3xl font-bold tracking-tight">
            How It Works
          </h2>
          <div className="grid gap-8 md:grid-cols-3">
            {steps.map((step, i) => (
              <div key={i} className="flex flex-col items-center gap-4 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-xl border border-border bg-card">
                  <step.icon className="h-6 w-6" />
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-muted-foreground">
                    {i + 1}.
                  </span>
                  <h3 className="font-semibold">{step.title}</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Separator />

      {/* Problem / Solution */}
      <section className="py-20 px-4">
        <div className="mx-auto grid max-w-4xl gap-12 md:grid-cols-2">
          <div className="space-y-4">
            <h3 className="text-xl font-semibold">The Problem</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              LLMs see SVG as raw XML — coordinates, path data, and attributes.
              They lack spatial understanding: what&apos;s left, right, aligned,
              symmetric, or proportional. They can parse syntax but not
              geometry.
            </p>
          </div>
          <div className="space-y-4">
            <h3 className="text-xl font-semibold">Our Solution</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              VectorSight runs 61 geometry transforms that extract spatial
              relationships, symmetry, topology, and visual hierarchy. The
              result is a ~1,200-token enrichment that any LLM can read — no
              training, no tools, just structured data.
            </p>
          </div>
        </div>
      </section>

      <Separator />

      {/* Stats */}
      <section className="py-20 px-4">
        <div className="mx-auto max-w-3xl">
          <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-3xl font-bold">{stat.value}</div>
                <div className="mt-1 text-sm text-muted-foreground">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Separator />

      {/* Hackathon Banner */}
      <section className="py-16 px-4 text-center">
        <Badge variant="outline" className="mb-4">
          Hackathon Entry
        </Badge>
        <h2 className="text-2xl font-bold tracking-tight">
          Built for The Strange Data Project
        </h2>
        <p className="mt-3 text-sm text-muted-foreground max-w-md mx-auto">
          VectorSight explores how structuring unconventional data formats
          can unlock new capabilities in language models.
        </p>
      </section>

      <Separator />

      {/* Team */}
      <section className="py-16 px-4 text-center">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Image
            src="/cybots.svg"
            alt="Cybots"
            width={32}
            height={32}
            className="dark:invert"
          />
          <h2 className="text-2xl font-bold tracking-tight">Team Cybots</h2>
        </div>
        <p className="text-sm text-muted-foreground">
          Exploring the intersection of geometry and language.
        </p>
      </section>

      <Separator />

      {/* Footer CTA */}
      <section className="py-16 px-4 text-center">
        <h2 className="text-2xl font-bold tracking-tight mb-4">
          Ready to see it in action?
        </h2>
        <Link
          href="/signup"
          className={buttonVariants({ size: "lg" })}
        >
          Try VectorSight
          <ArrowRight className="ml-2 h-4 w-4" />
        </Link>
      </section>

    </div>
  )
}

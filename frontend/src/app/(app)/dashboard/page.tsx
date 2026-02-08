import Link from "next/link"
import { ArrowRight, FileCode, MessageSquare, Settings } from "lucide-react"
import { Separator } from "@/components/ui/separator"

const quickActions = [
  {
    icon: FileCode,
    title: "Open Workspace",
    description: "Analyze and modify SVGs",
    href: "/workspace",
  },
  {
    icon: MessageSquare,
    title: "Chat with SVG",
    description: "Ask questions about your graphics",
    href: "/workspace",
  },
  {
    icon: Settings,
    title: "Settings",
    description: "Configure your account",
    href: "/settings",
  },
]

export default function DashboardPage() {
  return (
    <div className="mx-auto w-full max-w-4xl px-6 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Welcome to VectorSight. Get started with one of the actions below.
        </p>
      </div>

      <Separator />

      <div className="grid gap-4 sm:grid-cols-3">
        {quickActions.map((action) => (
          <Link
            key={action.title}
            href={action.href}
            className="flex flex-col gap-3 rounded-lg border border-border bg-card p-5 transition-colors hover:bg-accent"
          >
            <action.icon className="h-5 w-5 text-muted-foreground" />
            <div>
              <div className="text-sm font-medium">{action.title}</div>
              <div className="text-xs text-muted-foreground">
                {action.description}
              </div>
            </div>
            <ArrowRight className="mt-auto h-4 w-4 text-muted-foreground" />
          </Link>
        ))}
      </div>

      <Separator />

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Recent Activity</h2>
        <div className="rounded-lg border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
          No recent activity yet. Open the workspace to get started.
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Saved SVGs</h2>
        <div className="rounded-lg border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
          No saved SVGs. Analyze an SVG in the workspace to save it here.
        </div>
      </section>
    </div>
  )
}

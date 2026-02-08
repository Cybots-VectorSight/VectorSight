import { Check, ArrowRight } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"

const plans = [
  {
    name: "Free",
    description: "For individuals exploring VectorSight",
    features: [
      "5 SVG analyses per day",
      "Chat with SVG",
      "Basic modifications",
      "Sample SVGs",
      "Community support",
    ],
    cta: "Current Plan",
    highlighted: false,
    current: true,
  },
  {
    name: "Enterprise",
    description: "For teams and organizations",
    features: [
      "Unlimited analyses",
      "Priority LLM routing",
      "Batch processing API",
      "Custom transforms",
      "SSO & team management",
      "Dedicated support",
    ],
    cta: "Contact Us",
    highlighted: true,
    current: false,
  },
]

export default function PricingPage() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-10 space-y-10">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">Billing</h1>
        <p className="text-sm text-muted-foreground">
          Your current plan and upgrade options.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {plans.map((plan) => (
          <div
            key={plan.name}
            className={`flex flex-col rounded-xl border p-6 ${
              plan.highlighted
                ? "border-primary bg-card shadow-sm"
                : "border-border bg-card"
            }`}
          >
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <h2 className="text-xl font-semibold">{plan.name}</h2>
                {plan.current && <Badge variant="secondary">Current</Badge>}
                {plan.highlighted && <Badge>Popular</Badge>}
              </div>
              <p className="text-sm text-muted-foreground">
                {plan.description}
              </p>
            </div>

            <Separator className="my-5" />

            <ul className="flex-1 space-y-3">
              {plan.features.map((feature) => (
                <li key={feature} className="flex items-center gap-2 text-sm">
                  <Check className="h-4 w-4 shrink-0 text-muted-foreground" />
                  {feature}
                </li>
              ))}
            </ul>

            {plan.current ? (
              <div className="mt-6 flex h-10 items-center justify-center rounded-md border border-border text-sm text-muted-foreground">
                Current Plan
              </div>
            ) : (
              <a
                href="mailto:hello@vectorsight.dev"
                className={buttonVariants({
                  variant: "default",
                  className: "mt-6 w-full",
                })}
              >
                {plan.cta}
                <ArrowRight className="ml-2 h-4 w-4" />
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

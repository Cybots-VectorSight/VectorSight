import Link from "next/link"
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
    cta: "Get Started",
    href: "/signup",
    highlighted: false,
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
    href: "mailto:hello@vectorsight.dev",
    highlighted: true,
  },
]

export default function PricingPage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-16 space-y-10">
      <div className="space-y-3 text-center">
        <h1 className="text-3xl font-bold tracking-tight">Pricing</h1>
        <p className="text-sm text-muted-foreground">
          Start free. Scale when you need to.
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

            {plan.href.startsWith("mailto:") ? (
              <a
                href={plan.href}
                className={buttonVariants({
                  variant: "default",
                  className: "mt-6 w-full",
                })}
              >
                {plan.cta}
                <ArrowRight className="ml-2 h-4 w-4" />
              </a>
            ) : (
              <Link
                href={plan.href}
                className={buttonVariants({
                  variant: plan.highlighted ? "default" : "outline",
                  className: "mt-6 w-full",
                })}
              >
                {plan.cta}
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

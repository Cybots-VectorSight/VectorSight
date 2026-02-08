import Link from "next/link"
import Image from "next/image"

const legalLinks = [
  { href: "/privacy", label: "Privacy Policy" },
  { href: "/terms", label: "Terms of Service" },
  { href: "/cookies", label: "Cookie Policy" },
]

const productLinks = [
  { href: "/pricing", label: "Pricing" },
  { href: "/about", label: "About" },
]

export function PublicFooter() {
  return (
    <footer className="border-t border-border bg-card/50">
      <div className="mx-auto max-w-5xl px-4 py-10">
        <div className="grid gap-8 sm:grid-cols-3">
          {/* Brand */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Image
                src="/vectorsight-light.svg"
                alt="VectorSight"
                width={20}
                height={20}
                className="hidden dark:block"
              />
              <Image
                src="/vectorsight-dark.svg"
                alt="VectorSight"
                width={20}
                height={20}
                className="block dark:hidden"
              />
              <span className="text-sm font-semibold">VectorSight</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Transform SVG code into structured spatial data that LLMs can
              understand. By Team Cybots.
            </p>
          </div>

          {/* Product */}
          <div className="space-y-3">
            <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Product
            </h4>
            <ul className="space-y-2">
              {productLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-xs text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Legal */}
          <div className="space-y-3">
            <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Legal
            </h4>
            <ul className="space-y-2">
              {legalLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-xs text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-8 border-t border-border pt-6 flex flex-col items-center gap-2 sm:flex-row sm:justify-between">
          <p className="text-xs text-muted-foreground">
            &copy; {new Date().getFullYear()} VectorSight by Cybots. All rights
            reserved.
          </p>
          <p className="text-xs text-muted-foreground">
            The Strange Data Project
          </p>
        </div>
      </div>
    </footer>
  )
}

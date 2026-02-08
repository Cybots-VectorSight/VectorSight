import { Separator } from "@/components/ui/separator"

export default function CookiePolicyPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-16 space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Cookie Policy</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Last updated: February 2026
        </p>
      </div>

      <Separator />

      <div className="space-y-6 text-sm text-muted-foreground leading-relaxed">
        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            What Are Cookies
          </h2>
          <p>
            Cookies are small text files stored on your device when you visit
            a website. They help the site remember your preferences and
            improve your experience.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            Essential Cookies
          </h2>
          <p>
            These cookies are necessary for the Service to function properly.
            They include session management, authentication state, and theme
            preferences. These cannot be disabled.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            Analytics Cookies
          </h2>
          <p>
            We may use analytics cookies to understand how visitors interact
            with the Service. These cookies collect information anonymously
            and help us improve the user experience.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            Managing Cookies
          </h2>
          <p>
            You can manage your cookie preferences through the consent banner
            shown on your first visit, or through your browser settings. Note
            that disabling essential cookies may affect Service functionality.
          </p>
        </section>
      </div>
    </div>
  )
}

import { Separator } from "@/components/ui/separator"

export default function PrivacyPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-16 space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Privacy Policy</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Last updated: February 2026
        </p>
      </div>

      <Separator />

      <div className="space-y-6 text-sm text-muted-foreground leading-relaxed">
        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            1. Information We Collect
          </h2>
          <p>
            VectorSight collects minimal data to provide the Service. This
            includes account information (name, email) provided during
            registration and usage analytics to improve the experience.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            2. SVG Data Processing
          </h2>
          <p>
            SVG files uploaded to VectorSight are processed in real-time and
            are not permanently stored on our servers. Processed data is held
            in memory only for the duration of your active session.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            3. Cookies
          </h2>
          <p>
            We use essential cookies to maintain your session and preferences
            (such as theme selection). Analytics cookies may be used to
            understand usage patterns. You can manage cookie preferences via
            the cookie consent banner.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            4. Data Sharing
          </h2>
          <p>
            We do not sell, trade, or share your personal information with
            third parties except as required to operate the Service (e.g.,
            hosting infrastructure) or as required by law.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            5. Data Retention
          </h2>
          <p>
            Account data is retained for the duration of your account. You
            may request deletion of your account and associated data at any
            time by contacting us.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            6. Contact
          </h2>
          <p>
            For privacy-related inquiries, please reach out to Team Cybots
            through the project repository.
          </p>
        </section>
      </div>
    </div>
  )
}

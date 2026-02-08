import { Separator } from "@/components/ui/separator"

export default function TermsPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-16 space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          Terms of Service
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Last updated: February 2026
        </p>
      </div>

      <Separator />

      <div className="space-y-6 text-sm text-muted-foreground leading-relaxed">
        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            1. Acceptance of Terms
          </h2>
          <p>
            By accessing or using VectorSight (&quot;Service&quot;), you
            agree to be bound by these Terms of Service. If you do not agree
            to these terms, you may not use the Service.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            2. Description of Service
          </h2>
          <p>
            VectorSight is an experimental tool that transforms SVG code into
            structured spatial data for language model consumption. The
            Service is provided for demonstration and educational purposes as
            part of The Strange Data Project hackathon.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            3. User Accounts
          </h2>
          <p>
            You are responsible for maintaining the confidentiality of your
            account credentials and for all activities that occur under your
            account. You agree to notify us immediately of any unauthorized
            use.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            4. User Content
          </h2>
          <p>
            You retain ownership of all SVG files and content you submit to
            the Service. By using the Service, you grant us a limited license
            to process your content solely to provide the Service
            functionality.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            5. Acceptable Use
          </h2>
          <p>
            You agree not to: (a) upload malicious SVG files designed to
            exploit the Service; (b) attempt to access unauthorized areas of
            the system; (c) use the Service to violate any applicable laws;
            (d) interfere with the operation of the Service.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            6. Disclaimer of Warranties
          </h2>
          <p>
            The Service is provided &quot;as is&quot; and &quot;as
            available&quot; without warranties of any kind, either express or
            implied. We do not warrant that the Service will be
            uninterrupted, error-free, or secure.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            7. Limitation of Liability
          </h2>
          <p>
            In no event shall VectorSight or Team Cybots be liable for any
            indirect, incidental, special, or consequential damages arising
            from your use of the Service.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-foreground">
            8. Modifications
          </h2>
          <p>
            We reserve the right to modify these terms at any time. Changes
            will be effective immediately upon posting. Your continued use of
            the Service constitutes acceptance of the modified terms.
          </p>
        </section>
      </div>
    </div>
  )
}

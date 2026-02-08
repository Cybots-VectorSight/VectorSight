"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { ScrollText } from "lucide-react"
import { useAuth } from "@/lib/auth-context"

const TOS_KEY = "vectorsight-tos-accepted"

export function TosDialog() {
  const [visible, setVisible] = useState(false)
  const { isAuthenticated } = useAuth()

  useEffect(() => {
    if (isAuthenticated) {
      const accepted = localStorage.getItem(TOS_KEY)
      if (!accepted) setVisible(true)
    }
  }, [isAuthenticated])

  const accept = () => {
    localStorage.setItem(TOS_KEY, new Date().toISOString())
    setVisible(false)
  }

  if (!visible) return null

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <div className="mx-4 w-full max-w-lg rounded-lg border border-border bg-card shadow-lg">
        <div className="flex items-center gap-3 p-6 pb-2">
          <ScrollText className="h-5 w-5 text-muted-foreground" />
          <h2 className="text-lg font-semibold">Terms of Service</h2>
        </div>

        <div className="px-6 pb-4">
          <p className="text-sm text-muted-foreground">
            Please review and accept our terms to continue using VectorSight.
          </p>
        </div>

        <Separator />

        <div className="max-h-64 overflow-y-auto px-6 py-4 space-y-3 text-xs text-muted-foreground leading-relaxed">
          <p className="font-medium text-foreground">
            VectorSight Terms of Service
          </p>
          <p>
            Last updated: February 2026
          </p>
          <p>
            <strong className="text-foreground">1. Acceptance.</strong> By
            accessing or using VectorSight (&quot;Service&quot;), you agree to
            be bound by these Terms of Service. If you do not agree, do not
            use the Service.
          </p>
          <p>
            <strong className="text-foreground">2. Description.</strong>{" "}
            VectorSight is an experimental tool that transforms SVG code into
            structured spatial data for language model consumption. The
            Service is provided as-is for demonstration and educational
            purposes.
          </p>
          <p>
            <strong className="text-foreground">3. User Data.</strong> SVG
            files you upload are processed in-memory and are not stored on
            our servers beyond the duration of your session. We do not claim
            ownership of your content.
          </p>
          <p>
            <strong className="text-foreground">4. Acceptable Use.</strong>{" "}
            You agree not to use the Service to process malicious SVG files,
            attempt to exploit the underlying infrastructure, or violate any
            applicable laws.
          </p>
          <p>
            <strong className="text-foreground">5. Disclaimer.</strong> The
            Service is provided &quot;as is&quot; without warranties of any
            kind. VectorSight and Team Cybots are not liable for any damages
            arising from use of the Service.
          </p>
          <p>
            <strong className="text-foreground">6. Changes.</strong> We may
            update these terms at any time. Continued use constitutes
            acceptance of updated terms.
          </p>
          <p>
            <strong className="text-foreground">7. Contact.</strong> For
            questions about these terms, reach out to Team Cybots via the
            project repository.
          </p>
        </div>

        <Separator />

        <div className="flex justify-end gap-2 p-4">
          <Button size="sm" onClick={accept}>
            I Accept
          </Button>
        </div>
      </div>
    </div>
  )
}

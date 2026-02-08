"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { useAuth } from "@/lib/auth-context"
import { ScrollText, Shield, Cookie } from "lucide-react"

const TOS_KEY = "vectorsight-tos-accepted"
const COOKIE_KEY = "vectorsight-cookies"

export default function SettingsPage() {
  const { user } = useAuth()
  const [tosAccepted, setTosAccepted] = useState<string | null>(null)
  const [cookieChoice, setCookieChoice] = useState<string | null>(null)

  useEffect(() => {
    setTosAccepted(localStorage.getItem(TOS_KEY))
    setCookieChoice(localStorage.getItem(COOKIE_KEY))
  }, [])

  const resetCookies = () => {
    localStorage.removeItem(COOKIE_KEY)
    setCookieChoice(null)
    window.location.reload()
  }

  return (
    <div className="mx-auto w-full max-w-2xl px-6 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Manage your VectorSight preferences.
        </p>
      </div>

      <Separator />

      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Profile</h2>
        <div className="space-y-3">
          <div className="space-y-2">
            <Label htmlFor="display-name">Display Name</Label>
            <Input
              id="display-name"
              placeholder="Your name"
              defaultValue={user?.name ?? ""}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="you@example.com"
              defaultValue={user?.email ?? ""}
            />
          </div>
        </div>
      </section>

      <Separator />

      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Preferences</h2>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium">Baseline mode</div>
            <div className="text-xs text-muted-foreground">
              Show side-by-side comparison with/without enrichment
            </div>
          </div>
          <Switch />
        </div>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium">Auto-analyze</div>
            <div className="text-xs text-muted-foreground">
              Automatically analyze SVGs when loaded
            </div>
          </div>
          <Switch />
        </div>
      </section>

      <Separator />

      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Legal & Privacy</h2>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <ScrollText className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-sm font-medium">Terms of Service</div>
              <div className="text-xs text-muted-foreground">
                {tosAccepted
                  ? `Accepted on ${new Date(tosAccepted).toLocaleDateString()}`
                  : "Not yet accepted"}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {tosAccepted && (
              <Badge variant="secondary" className="text-xs">
                Accepted
              </Badge>
            )}
            <Link href="/terms">
              <Button variant="outline" size="sm">
                View
              </Button>
            </Link>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-sm font-medium">Privacy Policy</div>
              <div className="text-xs text-muted-foreground">
                How we handle your data
              </div>
            </div>
          </div>
          <Link href="/privacy">
            <Button variant="outline" size="sm">
              View
            </Button>
          </Link>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Cookie className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-sm font-medium">Cookie Preferences</div>
              <div className="text-xs text-muted-foreground">
                {cookieChoice
                  ? `Cookies ${cookieChoice}`
                  : "No preference set"}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {cookieChoice && (
              <Badge variant="secondary" className="text-xs capitalize">
                {cookieChoice}
              </Badge>
            )}
            <Button variant="outline" size="sm" onClick={resetCookies}>
              Reset
            </Button>
          </div>
        </div>
      </section>

      <Separator />

      <div className="flex justify-end">
        <Button>Save Changes</Button>
      </div>
    </div>
  )
}

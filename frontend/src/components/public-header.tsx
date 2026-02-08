"use client"

import Link from "next/link"
import Image from "next/image"
import { useTheme } from "next-themes"
import { usePathname, useRouter } from "next/navigation"
import { Sun, Moon, LogOut, LayoutDashboard } from "lucide-react"
import { Button } from "@/components/ui/button"
import { buttonVariants } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverPositioner,
} from "@/components/ui/popover"
import { useAuth } from "@/lib/auth-context"

const navLinks = [
  { href: "/pricing", label: "Pricing" },
  { href: "/about", label: "About" },
]

export function PublicHeader() {
  const { theme, setTheme } = useTheme()
  const pathname = usePathname()
  const router = useRouter()
  const { user, isAuthenticated, logout } = useAuth()

  const handleLogout = () => {
    logout()
    router.push("/")
  }

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-6">
        {/* Left — Logo */}
        <Link href="/" className="flex items-center gap-2">
          <Image
            src="/vectorsight-light.svg"
            alt="VectorSight"
            width={24}
            height={24}
            className="hidden dark:block"
          />
          <Image
            src="/vectorsight-dark.svg"
            alt="VectorSight"
            width={24}
            height={24}
            className="block dark:hidden"
          />
          <span className="text-sm font-semibold tracking-tight">
            VectorSight
          </span>
        </Link>

        {/* Center — Nav */}
        <nav className="absolute left-1/2 top-1/2 hidden -translate-x-1/2 -translate-y-1/2 items-center gap-1 sm:flex">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                pathname === link.href
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        {/* Right — Actions */}
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="h-8 w-8"
          >
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>

          {isAuthenticated && user ? (
            <Popover>
              <PopoverTrigger className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-xs font-medium transition-colors hover:bg-accent">
                {user.name[0]?.toUpperCase() ?? "?"}
              </PopoverTrigger>
              <PopoverPositioner side="bottom" align="end" sideOffset={8}>
                <PopoverContent className="w-56 p-1">
                  <div className="px-3 py-2">
                    <p className="text-sm font-medium">{user.name}</p>
                    <p className="text-xs text-muted-foreground">{user.email}</p>
                  </div>
                  <Separator className="my-1" />
                  <Link
                    href="/dashboard"
                    className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                  >
                    <LayoutDashboard className="h-4 w-4" />
                    Dashboard
                  </Link>
                  <Separator className="my-1" />
                  <button
                    onClick={handleLogout}
                    className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                  >
                    <LogOut className="h-4 w-4" />
                    Sign out
                  </button>
                </PopoverContent>
              </PopoverPositioner>
            </Popover>
          ) : (
            <>
              <Link
                href="/login"
                className={buttonVariants({ variant: "ghost", size: "sm" })}
              >
                Sign In
              </Link>
              <Link
                href="/signup"
                className={buttonVariants({ size: "sm" })}
              >
                Sign Up
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  )
}

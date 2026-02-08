"use client"

import Link from "next/link"
import Image from "next/image"
import { usePathname, useRouter } from "next/navigation"
import { useTheme } from "next-themes"
import {
  LayoutDashboard,
  PenTool,
  CreditCard,
  Settings,
  LogOut,
  Sun,
  Moon,
  ExternalLink,
  History,
  ChevronsUpDown,
} from "lucide-react"
import { useAuth } from "@/lib/auth-context"
import { Separator } from "@/components/ui/separator"
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverPositioner,
} from "@/components/ui/popover"

const mainNav = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/workspace", label: "Workspace", icon: PenTool },
  { href: "/history", label: "History", icon: History },
]

const bottomNav = [
  { href: "/billing", label: "Billing", icon: CreditCard },
  { href: "/settings", label: "Settings", icon: Settings },
]

function NavLink({
  href,
  label,
  icon: Icon,
  isActive,
}: {
  href: string
  label: string
  icon: React.ComponentType<{ className?: string }>
  isActive: boolean
}) {
  return (
    <Link
      href={href}
      className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
        isActive
          ? "bg-accent text-accent-foreground font-medium"
          : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
      }`}
    >
      <Icon className="h-4 w-4 shrink-0" />
      {label}
    </Link>
  )
}

export function AppSidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { user, logout } = useAuth()
  const { theme, setTheme } = useTheme()

  const handleLogout = () => {
    logout()
    router.push("/")
  }

  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-border bg-card">
      {/* Logo */}
      <div className="flex h-14 items-center px-4">
        <Link href="/dashboard" className="flex items-center gap-2">
          <Image
            src="/vectorsight-light.svg"
            alt="VectorSight"
            width={22}
            height={22}
            className="hidden dark:block"
          />
          <Image
            src="/vectorsight-dark.svg"
            alt="VectorSight"
            width={22}
            height={22}
            className="block dark:hidden"
          />
          <span className="text-sm font-semibold tracking-tight">
            VectorSight
          </span>
        </Link>
      </div>

      <Separator />

      {/* Main nav */}
      <nav className="space-y-1 px-2 py-3">
        {mainNav.map((item) => (
          <NavLink
            key={item.href}
            {...item}
            isActive={pathname === item.href}
          />
        ))}
      </nav>

      {/* Spacer */}
      <div className="flex-1" />

      <Separator />

      {/* Bottom nav */}
      <nav className="space-y-1 px-2 py-2">
        {bottomNav.map((item) => (
          <NavLink
            key={item.href}
            {...item}
            isActive={pathname === item.href}
          />
        ))}
      </nav>

      <Separator />

      {/* User section with popover */}
      <div className="p-2">
        <Popover>
          <PopoverTrigger
            className="flex w-full items-center gap-2.5 rounded-md px-2 py-2 transition-colors hover:bg-accent/50"
          >
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium">
              {user?.name?.[0]?.toUpperCase() ?? "?"}
            </div>
            <div className="min-w-0 flex-1 text-left">
              <p className="truncate text-sm font-medium">{user?.name}</p>
              <p className="truncate text-xs text-muted-foreground">
                {user?.email}
              </p>
            </div>
            <ChevronsUpDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          </PopoverTrigger>
          <PopoverPositioner side="top" align="start" sideOffset={8}>
            <PopoverContent className="w-56 p-1">
              <button
                onClick={() =>
                  setTheme(theme === "dark" ? "light" : "dark")
                }
                className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <Sun className="h-4 w-4 dark:hidden" />
                <Moon className="hidden h-4 w-4 dark:block" />
                {theme === "dark" ? "Light mode" : "Dark mode"}
              </button>
              <Separator className="my-1" />
              <Link
                href="/"
                className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <ExternalLink className="h-4 w-4" />
                Visit Website
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
      </div>
    </aside>
  )
}

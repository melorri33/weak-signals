import { useQuery } from "@tanstack/react-query"
import { MoonIcon, SunIcon } from "lucide-react"
import { Link, Outlet, ScrollRestoration } from "react-router"

import { api } from "@/api/client"
import { useTheme } from "@/components/theme-provider"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

export function AppShell() {
  return (
    <div className="flex min-h-svh flex-col">
      <a
        href="#main"
        className="sr-only z-50 rounded-md bg-primary px-3 py-2 text-primary-foreground focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
      >
        К содержимому
      </a>
      <header data-print="hide" className="border-b bg-card">
        <div className="mx-auto flex h-14 w-full max-w-6xl items-center gap-4 px-4 sm:px-6">
          <Link
            to="/"
            className="flex items-center gap-2 font-semibold tracking-tight"
          >
            <SignalMark />
            Слабые сигналы
          </Link>
          <div className="ml-auto flex items-center gap-2">
            <HealthStatus />
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main
        id="main"
        tabIndex={-1}
        className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 outline-none sm:px-6 sm:py-10"
      >
        <Outlet />
      </main>
      <ScrollRestoration />
    </div>
  )
}

/** Знак: слабый всплеск над шумом — ровно то, что ищет сервис. */
function SignalMark() {
  return (
    <svg viewBox="0 0 24 24" className="size-6 text-primary" aria-hidden="true">
      <path
        d="M2 15h4l1.5-2 1.5 2h2l2-9 2 13 1.5-4H22"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function HealthStatus() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
  })
  if (isError) {
    return (
      <StatusDot
        ok={false}
        label="Сервер не отвечает"
        detail="Запустите api: docker compose up -d api"
      />
    )
  }
  if (!data) return null
  const detail = data.notes?.length
    ? data.notes.join(". ")
    : "Модель и база доступны"
  return (
    <StatusDot
      ok={data.llm_available}
      label={
        data.llm_available
          ? `Модель ${data.llm_model}`
          : `Модель ${data.llm_model} недоступна`
      }
      detail={detail}
    />
  )
}

function StatusDot({
  ok,
  label,
  detail,
}: {
  ok: boolean
  label: string
  detail: string
}) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <span
            tabIndex={0}
            className="hidden items-center gap-2 rounded-md px-2 py-1 text-xs text-muted-foreground sm:inline-flex"
          />
        }
      >
        <span
          className={cn(
            "size-2 rounded-full",
            ok ? "bg-level-high" : "bg-level-low"
          )}
          aria-hidden="true"
        />
        {label}
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-72">
        {detail}
      </TooltipContent>
    </Tooltip>
  )
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const isDark =
    theme === "dark" ||
    (theme === "system" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches)
  const label = isDark ? "Светлая тема" : "Тёмная тема"
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label={label}
    >
      {isDark ? <SunIcon /> : <MoonIcon />}
    </Button>
  )
}

import { useId } from "react"
import { CornerDownRightIcon } from "lucide-react"

import type { SearchResult } from "@/api/client"
import { CONFIDENT_THRESHOLD, formatNumber, percent } from "@/lib/format"
import { cn } from "@/lib/utils"

export type RunTab = "top" | "excluded" | "candidates" | "method"

type Step = {
  value: number
  label: string
  short: string
  note: string
  tab: RunTab
  drop?: { value: number; text: string; tab: RunTab }
  emphasis?: boolean
}

const WIDTH = 1000
const HEIGHT = 120
const MAX_HALF = 46
const MIN_HALF = 3

/**
 * Воронка отсева: от собранных документов до уверенных сигналов — одной сужающейся полосой.
 * Это и есть метод: слабый сигнал отделяется от шума по шагам. И заодно три «плюса» из ТЗ:
 * число обработанных источников, число кандидатов со ссылкой на список, число сигналов с уверенностью >75%.
 * Толщина полосы — в логарифмической шкале, чтобы 500 документов и 3 сигнала помещались рядом.
 * Под шагами — сколько ушло и почему: клик ведёт к списку отсеянных.
 */
export function Funnel({
  result,
  onSelect,
}: {
  result: SearchResult
  onSelect: (tab: RunTab) => void
}) {
  const gradientId = useId()
  const excluded = result.excluded?.filter((d) => d.excluded).length ?? 0
  const scored = result.scored?.length ?? 0
  const top = result.top?.length ?? 0
  const steps: Step[] = [
    {
      value: result.documents_processed ?? 0,
      label: "Обработано источников",
      short: "документов",
      note: "статьи, препринты, новости",
      tab: "method",
    },
    {
      value: result.candidates_found ?? 0,
      label: "Технологий-кандидатов",
      short: "кандидатов",
      note: "выписано из документов",
      tab: "candidates",
    },
    {
      value: scored,
      label: "Прошли отсев",
      short: "прошли отсев",
      note: "оценены моделью",
      tab: "candidates",
      drop: excluded
        ? {
            value: excluded,
            text: "отсеяно правилами: зрелое, хайп, шум",
            tab: "excluded",
          }
        : undefined,
    },
    {
      value: top,
      label: "В выдаче",
      short: "в выдаче",
      note: "с карточкой и источниками",
      tab: "top",
      drop:
        scored > top
          ? {
              value: scored - top,
              text: "ниже топ-15 или без подтверждений",
              tab: "candidates",
            }
          : undefined,
    },
    {
      value: result.confident_signals ?? 0,
      label: "Уверенных сигналов",
      short: "уверенных",
      note: `оценка модели выше ${percent(CONFIDENT_THRESHOLD)}`,
      tab: "top",
      emphasis: true,
    },
  ]
  const max = Math.max(1, ...steps.map((s) => s.value))
  const halves = steps.map(
    (s) =>
      MIN_HALF + (MAX_HALF - MIN_HALF) * (Math.log1p(s.value) / Math.log1p(max))
  )

  return (
    <section
      aria-label="Воронка отсева"
      className="overflow-hidden rounded-xl border bg-card"
    >
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        className="funnel-reveal block h-20 w-full sm:h-28"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" style={{ stopColor: "var(--conf-low)" }} />
            <stop offset="1" style={{ stopColor: "var(--conf-high)" }} />
          </linearGradient>
        </defs>
        <path d={bandPath(halves)} fill={`url(#${gradientId})`} />
      </svg>
      <ol className="grid grid-cols-5">
        {steps.map((step) => (
          <li
            key={step.label}
            className="flex min-w-0 flex-col border-r last:border-r-0"
          >
            <button
              type="button"
              onClick={() => onSelect(step.tab)}
              className={cn(
                "flex w-full flex-1 flex-col gap-1 px-2 pt-3 pb-2 text-left transition-colors hover:bg-accent/60 focus-visible:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none focus-visible:ring-inset sm:px-4",
                step.emphasis && "bg-accent/50"
              )}
              aria-label={`${step.label}: ${step.value}. Показать список`}
            >
              <span
                className={cn(
                  "text-lg leading-none font-semibold tracking-tight sm:text-3xl",
                  step.emphasis && "text-primary"
                )}
              >
                {formatNumber(step.value)}
              </span>
              <span className="text-[0.6875rem] leading-snug font-medium hyphens-auto sm:text-sm">
                <span className="sm:hidden">{step.short}</span>
                <span className="hidden sm:inline">{step.label}</span>
              </span>
              <span className="hidden text-xs leading-snug text-muted-foreground sm:block">
                {step.note}
              </span>
            </button>
            {step.drop ? (
              <button
                type="button"
                onClick={() => onSelect(step.drop!.tab)}
                className="flex items-start gap-1 px-2 pb-3 text-left text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none sm:px-4"
              >
                <CornerDownRightIcon
                  className="mt-px size-3.5 shrink-0"
                  aria-hidden="true"
                />
                <span>
                  <span className="sm:hidden">−</span>
                  {step.drop.value}
                  <span className="hidden sm:inline"> {step.drop.text}</span>
                </span>
              </button>
            ) : (
              <span className="pb-3" />
            )}
          </li>
        ))}
      </ol>
    </section>
  )
}

/** Полоса через центры колонок: плоско в колонке, плавный изгиб между ними. */
function bandPath(halves: number[]): string {
  const mid = HEIGHT / 2
  const col = WIDTH / halves.length
  const top: string[] = []
  const bottom: string[] = []
  halves.forEach((h, i) => {
    const left = i * col + col * 0.2
    const right = (i + 1) * col - col * 0.2
    if (i === 0) {
      top.push(`M0 ${mid - h}`, `L${right} ${mid - h}`)
    } else {
      const prevRight = i * col - col * 0.2
      const c = (prevRight + left) / 2
      top.push(
        `C${c} ${mid - halves[i - 1]} ${c} ${mid - h} ${left} ${mid - h}`,
        `L${i === halves.length - 1 ? WIDTH : right} ${mid - h}`
      )
    }
  })
  for (let i = halves.length - 1; i >= 0; i--) {
    const h = halves[i]
    const left = i * col + col * 0.2
    if (i === halves.length - 1) {
      bottom.push(`L${WIDTH} ${mid + h}`, `L${left} ${mid + h}`)
    } else {
      const right = (i + 1) * col - col * 0.2
      const nextLeft = (i + 1) * col + col * 0.2
      const c = (right + nextLeft) / 2
      bottom.push(
        `C${c} ${mid + halves[i + 1]} ${c} ${mid + h} ${right} ${mid + h}`,
        `L${i === 0 ? 0 : left} ${mid + h}`
      )
    }
  }
  return `${top.join(" ")} ${bottom.join(" ")} Z`
}

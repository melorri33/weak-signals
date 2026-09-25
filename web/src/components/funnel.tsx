import type { SearchResult } from "@/api/client"
import { CONFIDENT_THRESHOLD, formatNumber, percent } from "@/lib/format"
import { cn } from "@/lib/utils"

export type RunTab = "top" | "excluded" | "candidates" | "method"

type Step = {
  value: number
  label: string
  note: string
  tab?: RunTab
  emphasis?: boolean
}

/**
 * Воронка отсева: от собранных документов до уверенных сигналов.
 * Это и есть метод — слабый сигнал отделяется от шума по шагам, — и заодно три «плюса» из ТЗ:
 * число обработанных источников, число кандидатов со ссылкой на список, число сигналов с уверенностью >75%.
 * Полоса под числом — в логарифмической шкале, чтобы 2000 документов и 4 сигнала помещались рядом.
 */
export function Funnel({
  result,
  onSelect,
}: {
  result: SearchResult
  onSelect: (tab: RunTab) => void
}) {
  const excluded = result.excluded?.length ?? 0
  const scored = result.scored?.length ?? 0
  const top = result.top?.length ?? 0
  const steps: Step[] = [
    {
      value: result.documents_processed ?? 0,
      label: "Обработано источников",
      note: "статьи, препринты, новости",
      tab: "method",
    },
    {
      value: result.candidates_found ?? 0,
      label: "Технологий-кандидатов",
      note: "выписано из документов",
      tab: "candidates",
    },
    {
      value: excluded,
      label: "Отсеяно правилами",
      note: "зрелое, хайп, шум",
      tab: "excluded",
    },
    {
      value: scored,
      label: "Оценено моделью",
      note: "уверенность и объяснение",
      tab: "candidates",
    },
    {
      value: top,
      label: "В топ-15",
      note: "с карточкой и источниками",
      tab: "top",
    },
    {
      value: result.confident_signals ?? 0,
      label: "Уверенных сигналов",
      note: `оценка модели выше ${percent(CONFIDENT_THRESHOLD)}`,
      tab: "top",
      emphasis: true,
    },
  ]
  const max = Math.max(1, ...steps.map((s) => s.value))

  return (
    <ol className="grid grid-cols-2 overflow-hidden rounded-xl border bg-card sm:grid-cols-3 lg:grid-cols-6">
      {steps.map((step, i) => (
        <li
          key={step.label}
          className="border-r border-b last:border-r-0 lg:border-b-0"
        >
          <button
            type="button"
            onClick={() => step.tab && onSelect(step.tab)}
            className={cn(
              "flex h-full w-full flex-col gap-2 p-4 text-left transition-colors hover:bg-accent/60 focus-visible:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none focus-visible:ring-inset",
              step.emphasis && "bg-accent/50"
            )}
            aria-label={`${step.label}: ${step.value}. Показать список`}
          >
            <span
              className={cn(
                "text-3xl leading-none font-semibold tracking-tight",
                step.emphasis ? "text-primary" : "text-foreground"
              )}
            >
              {formatNumber(step.value)}
            </span>
            <span className="text-sm leading-snug font-medium">
              {step.label}
            </span>
            <span className="text-xs leading-snug text-muted-foreground">
              {step.note}
            </span>
            <span
              className="mt-auto h-1 w-full rounded-full bg-muted"
              aria-hidden="true"
            >
              <span
                className={cn(
                  "block h-full rounded-full",
                  i === steps.length - 1 ? "bg-primary" : "bg-chart-2"
                )}
                style={{
                  width: `${Math.max(3, (Math.log1p(step.value) / Math.log1p(max)) * 100)}%`,
                }}
              />
            </span>
          </button>
        </li>
      ))}
    </ol>
  )
}

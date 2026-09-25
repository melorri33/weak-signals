import { useEffect, useState } from "react"
import { CheckIcon } from "lucide-react"

import type { SearchResult } from "@/api/client"
import { Spinner } from "@/components/ui/spinner"
import {
  PIPELINE_STAGES,
  formatDuration,
  formatNumber,
  plural,
  stageIndex,
} from "@/lib/format"
import { cn } from "@/lib/utils"

/** Прогресс прогона по шагам конвейера. Процента нет и не выдумываем: шаги идут разное время. */
export function StageProgress({ result }: { result: SearchResult }) {
  const current = stageIndex(result.stage ?? "")
  const elapsed = useElapsed(result.started_at ?? null)
  const phrases = result.expanded_phrases ?? []

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_16rem]">
      <div className="flex min-w-0 flex-col gap-8">
        <ol className="flex flex-col" aria-label="Шаги поиска">
          {PIPELINE_STAGES.map((step, i) => {
            const state =
              current === -1
                ? "pending"
                : i < current
                  ? "done"
                  : i === current
                    ? "active"
                    : "pending"
            return (
              <li
                key={step.stage}
                className="relative flex gap-4 pb-6 last:pb-0"
                aria-current={state === "active" ? "step" : undefined}
              >
                {i < PIPELINE_STAGES.length - 1 ? (
                  <span
                    className={cn(
                      "absolute top-7 left-3.5 h-[calc(100%-1.75rem)] w-px",
                      state === "done" ? "bg-primary" : "bg-border"
                    )}
                    aria-hidden="true"
                  />
                ) : null}
                <span
                  className={cn(
                    "relative z-10 flex size-7 shrink-0 items-center justify-center rounded-full border text-xs",
                    state === "done" &&
                      "border-primary bg-primary text-primary-foreground",
                    state === "active" && "border-primary bg-card text-primary",
                    state === "pending" && "bg-card text-muted-foreground"
                  )}
                >
                  {state === "done" ? (
                    <CheckIcon className="size-4" />
                  ) : state === "active" ? (
                    <Spinner />
                  ) : (
                    i + 1
                  )}
                </span>
                <div className="flex flex-col gap-0.5 pt-0.5">
                  <span
                    className={cn(
                      "font-medium",
                      state === "pending" && "text-muted-foreground"
                    )}
                  >
                    {step.title}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {step.detail}
                  </span>
                </div>
              </li>
            )
          })}
        </ol>
        {phrases.length > 0 ? <Phrases phrases={phrases} /> : null}
      </div>
      <dl className="grid h-fit grid-cols-3 gap-4 rounded-xl border bg-card p-4 lg:grid-cols-1">
        <Counter label="Прошло" value={formatDuration(elapsed)} />
        <Counter
          label="Документов"
          value={
            result.documents_processed
              ? formatNumber(result.documents_processed)
              : "—"
          }
        />
        <Counter
          label="Кандидатов"
          value={
            result.candidates_found
              ? formatNumber(result.candidates_found)
              : "—"
          }
        />
      </dl>
    </div>
  )
}

/**
 * Фразы, которые модель вывела из запроса, — первое, что видно из работы системы: 16 минут ожидания
 * выглядят как поиск, а не как зависание. Список технологий заранее не задан — это тоже видно.
 */
function Phrases({ phrases }: { phrases: string[] }) {
  return (
    <section className="flex flex-col gap-3" aria-labelledby="live-phrases">
      <h2 id="live-phrases" className="text-sm font-medium">
        Ищем по {phrases.length}{" "}
        {plural(phrases.length, "фразе", "фразам", "фразам")}
      </h2>
      <ul className="flex flex-wrap gap-2">
        {phrases.map((phrase, i) => (
          <li
            key={phrase}
            className="animate-in rounded-md border bg-card px-2.5 py-1 text-sm duration-500 fill-mode-both fade-in slide-in-from-bottom-1"
            style={{ animationDelay: `${i * 60}ms` }}
            translate="no"
          >
            {phrase}
          </li>
        ))}
      </ul>
    </section>
  )
}

function Counter({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="text-xl font-semibold" aria-live="polite">
        {value}
      </dd>
    </div>
  )
}

function useElapsed(startedAt: string | null): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])
  return startedAt
    ? Math.max(0, (now - new Date(startedAt).getTime()) / 1000)
    : 0
}

import { useEffect, useState } from "react"
import { CheckIcon } from "lucide-react"

import type { SearchResult } from "@/api/client"
import { Spinner } from "@/components/ui/spinner"
import {
  PIPELINE_STAGES,
  formatDuration,
  formatNumber,
  stageIndex,
} from "@/lib/format"
import { cn } from "@/lib/utils"

/** Прогресс прогона по шагам конвейера. Процента нет и не выдумываем: шаги идут разное время. */
export function StageProgress({ result }: { result: SearchResult }) {
  const current = stageIndex(result.stage ?? "")
  const elapsed = useElapsed(result.started_at ?? null)

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_16rem]">
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

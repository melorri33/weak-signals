import { ArrowDownIcon, ArrowUpIcon } from "lucide-react"

import type { Explanation } from "@/api/client"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

/**
 * Вклады признаков в оценку (SHAP): вправо от нуля — за слабый сигнал, влево — против.
 * Полярность передаётся трижды — стороной от оси, стрелкой и словом в подсказке, — поэтому цвет
 * не единственный носитель смысла. Подпись — токеном текста, а не цветом столбика.
 */
export function ShapBars({ reasons }: { reasons: Explanation[] }) {
  const sorted = [...reasons].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
  )
  const max = Math.max(1e-6, ...sorted.map((r) => Math.abs(r.contribution)))

  if (sorted.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Модель не вернула разложение оценки для этой технологии.
      </p>
    )
  }

  return (
    <figure className="flex flex-col gap-3">
      <div
        className="grid grid-cols-1 items-center gap-x-4 text-xs text-muted-foreground sm:grid-cols-[1fr_minmax(8rem,14rem)]"
        aria-hidden="true"
      >
        <span className="hidden sm:block" />
        <span className="grid grid-cols-2">
          <span className="pr-2 text-right">против</span>
          <span className="pl-2">за сигнал</span>
        </span>
      </div>
      <ul className="flex flex-col gap-2.5">
        {sorted.map((reason) => {
          const positive = reason.contribution >= 0
          const width = `${(Math.abs(reason.contribution) / max) * 100}%`
          const signed = `${positive ? "+" : "−"}${Math.abs(reason.contribution).toFixed(2)}`
          return (
            <li
              key={reason.feature}
              className="grid grid-cols-1 items-center gap-x-4 gap-y-1 sm:grid-cols-[1fr_minmax(8rem,14rem)]"
            >
              <span className="flex gap-1.5 text-sm">
                {positive ? (
                  <ArrowUpIcon
                    className="mt-0.5 size-4 shrink-0 text-shap-for"
                    aria-hidden="true"
                  />
                ) : (
                  <ArrowDownIcon
                    className="mt-0.5 size-4 shrink-0 text-shap-against"
                    aria-hidden="true"
                  />
                )}
                <span>
                  {reason.text}
                  <span className="sr-only">
                    {positive ? ", за слабый сигнал" : ", против"}, вклад{" "}
                    {signed}
                  </span>
                </span>
              </span>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <div
                      className="relative grid h-6 grid-cols-2 items-center"
                      tabIndex={-1}
                      aria-hidden="true"
                    />
                  }
                >
                  <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
                  <span className="flex h-full items-center justify-end pr-px">
                    {!positive ? (
                      <Bar
                        width={width}
                        className="rounded-l bg-shap-against"
                      />
                    ) : null}
                  </span>
                  <span className="flex h-full items-center pl-px">
                    {positive ? (
                      <Bar width={width} className="rounded-r bg-shap-for" />
                    ) : null}
                  </span>
                </TooltipTrigger>
                <TooltipContent className="flex-col items-start gap-0.5">
                  <span>
                    Вклад {signed} {positive ? "за слабый сигнал" : "против"}
                  </span>
                  <span className="opacity-80">
                    Признак {reason.feature}
                    {reason.value !== null && reason.value !== undefined
                      ? ` = ${formatValue(reason.value)}`
                      : ""}
                  </span>
                </TooltipContent>
              </Tooltip>
            </li>
          )
        })}
      </ul>
      <figcaption className="text-xs text-muted-foreground">
        Длина полосы — насколько признак сдвинул оценку классификатора (SHAP).
        Наведите на полосу, чтобы увидеть признак и его значение.
      </figcaption>
    </figure>
  )
}

function Bar({ width, className }: { width: string; className: string }) {
  return (
    <span className={cn("block h-3 min-w-1", className)} style={{ width }} />
  )
}

function formatValue(value: number | string): string {
  if (typeof value === "string") return value
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}

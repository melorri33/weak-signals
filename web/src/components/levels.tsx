import { Badge } from "@/components/ui/badge"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import type { TrustLevel } from "@/api/client"
import {
  CONFIDENCE_LABEL,
  TRUST_CRITERIA,
  TRUST_LABEL,
  TRUST_LEVEL,
  confidenceLevel,
  percent,
  type Level,
} from "@/lib/format"
import { cn } from "@/lib/utils"

/** Доверие к источнику — статус: светофор. */
const BAR_COLOR: Record<Level, string> = {
  high: "bg-level-high",
  mid: "bg-level-mid",
  low: "bg-level-low",
}

/** Уверенность — величина: один тон разной густоты. */
const CONF_COLOR: Record<Level, string> = {
  high: "bg-conf-high",
  mid: "bg-conf-mid",
  low: "bg-conf-low",
}

/** Уверенность модели: словом и процентом — цвет не единственный носитель смысла. */
export function ConfidenceBadge({ score }: { score: number }) {
  const level = confidenceLevel(score)
  return (
    <Badge variant="outline" className="gap-1.5 bg-card">
      <span
        className={cn("size-2 rounded-full", CONF_COLOR[level])}
        aria-hidden="true"
      />
      {CONFIDENCE_LABEL[level]} ({percent(score)})
    </Badge>
  )
}

/** Уверенность в строке выдачи: крупная цифра — главное число строки, под ней шкала и слово. */
export function ConfidenceScore({ score }: { score: number }) {
  const level = confidenceLevel(score)
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-2xl leading-none font-semibold tracking-tight">
          {percent(score)}
        </span>
        <span className="text-xs text-muted-foreground">
          {CONFIDENCE_LABEL[level].toLowerCase()}
        </span>
      </div>
      <ConfidenceBar score={score} className="max-w-32" />
    </div>
  )
}

export function ConfidenceBar({
  score,
  className,
}: {
  score: number
  className?: string
}) {
  const level = confidenceLevel(score)
  return (
    <div
      className={cn(
        "h-1.5 w-full overflow-hidden rounded-full bg-muted",
        className
      )}
      role="meter"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(score * 100)}
      aria-label="Уверенность модели"
    >
      <div
        className={cn("h-full rounded-full", CONF_COLOR[level])}
        style={{ width: percent(score) }}
      />
    </div>
  )
}

/** Уровень доверия с критерием во всплывающей подсказке: ТЗ просит показывать «уровень или критерии». */
export function TrustBadge({ trust }: { trust: TrustLevel }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={<Badge variant={TRUST_LEVEL[trust]} tabIndex={0} />}
      >
        {TRUST_LABEL[trust]}
      </TooltipTrigger>
      <TooltipContent className="max-w-64">
        {TRUST_CRITERIA[trust]}
      </TooltipContent>
    </Tooltip>
  )
}

/** Точки доверия по источникам карточки: сколько высоких, средних и низких — видно без клика. */
export function TrustDots({ trusts }: { trusts: TrustLevel[] }) {
  const order: TrustLevel[] = ["high", "medium", "low"]
  const sorted = [...trusts].sort((a, b) => order.indexOf(a) - order.indexOf(b))
  const counts = order
    .map((t) => ({ t, n: trusts.filter((x) => x === t).length }))
    .filter((c) => c.n > 0)
    .map((c) => `${TRUST_LABEL[c.t].toLowerCase()}: ${c.n}`)
    .join(", ")
  return (
    <span
      className="inline-flex items-center gap-1"
      aria-label={counts}
      title={counts}
    >
      {sorted.map((t, i) => (
        <span
          key={i}
          className={cn("size-2 rounded-full", BAR_COLOR[TRUST_LEVEL[t]])}
        />
      ))}
    </span>
  )
}

/** Состав источников по доверию одной полосой: доля низкого доверия видна сразу. */
export function TrustComposition({ trusts }: { trusts: TrustLevel[] }) {
  const order: TrustLevel[] = ["high", "medium", "low"]
  const parts = order
    .map((trust) => ({ trust, n: trusts.filter((t) => t === trust).length }))
    .filter((p) => p.n > 0)
  if (parts.length === 0) return null
  return (
    <div className="flex flex-col gap-2.5">
      <div
        className="flex h-2.5 gap-0.5 overflow-hidden rounded-full"
        aria-hidden="true"
      >
        {parts.map(({ trust, n }) => (
          <span
            key={trust}
            className={cn("h-full", BAR_COLOR[TRUST_LEVEL[trust]])}
            style={{ flexGrow: n }}
          />
        ))}
      </div>
      <ul className="flex flex-col gap-2">
        {parts.map(({ trust, n }) => (
          <li
            key={trust}
            className="flex items-center justify-between gap-3 text-sm"
          >
            <TrustBadge trust={trust} />
            <span>{n}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

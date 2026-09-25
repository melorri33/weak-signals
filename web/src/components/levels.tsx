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

const BAR_COLOR: Record<Level, string> = {
  high: "bg-level-high",
  mid: "bg-level-mid",
  low: "bg-level-low",
}

/** Уверенность модели: словом и процентом — цвет не единственный носитель смысла. */
export function ConfidenceBadge({ score }: { score: number }) {
  const level = confidenceLevel(score)
  return (
    <Badge variant={level}>
      {CONFIDENCE_LABEL[level]} ({percent(score)})
    </Badge>
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
        className={cn("h-full rounded-full", BAR_COLOR[level])}
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

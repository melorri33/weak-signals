import type { CandidateEvidence } from "@/api/client"
import { recentShare, yearSeries } from "@/lib/evidence"
import { formatNumber, percent, plural } from "@/lib/format"
import { cn } from "@/lib/utils"

const SPARK_YEARS = 8
const CHART_YEARS = 12

function lastYears(
  byYear: Record<string, number> | null | undefined,
  count: number
) {
  const series = yearSeries(byYear)
  const now = new Date().getFullYear()
  return Array.from({ length: count }, (_, i) => {
    const year = now - count + 1 + i
    return { year, n: series.get(year) ?? 0, partial: year === now }
  })
}

/**
 * Публикации по годам в строке выдачи: растущий хвост объясняет «почему ранняя» без слов.
 * Своя шкала у каждой строки — важна форма; всего публикаций подписано рядом.
 */
export function PubSparkline({ item }: { item: CandidateEvidence }) {
  const years = lastYears(item.pubs_by_year, SPARK_YEARS)
  const max = Math.max(1, ...years.map((y) => y.n))
  const total = item.features.total_pubs
  const summary = years.map((y) => `${y.year}: ${y.n}`).join(", ")
  return (
    <div className="flex items-end gap-2" title={summary}>
      <div
        className="flex h-6 items-end gap-0.5"
        role="img"
        aria-label={`Публикации по годам: ${summary}`}
      >
        {years.map((y) => (
          <span
            key={y.year}
            className={cn(
              "w-1.5 rounded-t-[2px]",
              y.partial ? "bg-conf-low" : "bg-conf-mid"
            )}
            style={{ height: `${Math.max(y.n ? 12 : 4, (y.n / max) * 100)}%` }}
          />
        ))}
      </div>
      {total !== null && total !== undefined ? (
        <span className="text-xs text-muted-foreground">
          {formatNumber(total)}
        </span>
      ) : null}
    </div>
  )
}

/**
 * Динамика в инсайте: публикации по годам и измеренные признаки рядом.
 * Текущий год светлее и подписан — он неполный, спад в нём не спад.
 */
export function PubDynamics({ item }: { item: CandidateEvidence }) {
  const years = lastYears(item.pubs_by_year, CHART_YEARS)
  const max = Math.max(1, ...years.map((y) => y.n))
  const peak = years.reduce((a, b) => (b.n > a.n ? b : a), years[0])
  const f = item.features
  const recent = recentShare(item)
  const facts = [
    f.total_pubs !== null && f.total_pubs !== undefined
      ? ["Публикаций всего", formatNumber(f.total_pubs)]
      : null,
    f.first_seen_year ? ["Первая публикация", `${f.first_seen_year} г.`] : null,
    recent !== null ? ["Вышло за последние 3 года", percent(recent)] : null,
    typeof f.growth_3y === "number"
      ? [
          "Рост в год за 3 года",
          `${f.growth_3y >= 0 ? "+" : ""}${percent(f.growth_3y)}`,
        ]
      : null,
    typeof f.extra?.preprint_share === "number"
      ? ["Доля препринтов", percent(f.extra.preprint_share)]
      : null,
    f.news_total !== null && f.news_total !== undefined
      ? ["Новостей в техмедиа", formatNumber(f.news_total)]
      : null,
    f.has_wikipedia !== null && f.has_wikipedia !== undefined
      ? ["Статья в Википедии", f.has_wikipedia ? "есть" : "нет"]
      : null,
  ].filter(Boolean) as [string, string][]

  return (
    <div className="flex flex-col gap-6">
      {max > 1 || years.some((y) => y.n > 0) ? (
        <figure className="flex flex-col gap-2">
          <div
            className="grid h-40 items-end gap-1 border-b"
            style={{
              gridTemplateColumns: `repeat(${years.length}, minmax(0, 1fr))`,
            }}
            role="img"
            aria-label={`Публикации по годам: ${years.map((y) => `${y.year} — ${y.n}`).join(", ")}`}
          >
            {years.map((y) => (
              <div
                key={y.year}
                className="flex h-full flex-col items-center justify-end gap-1"
                title={`${y.year}: ${formatNumber(y.n)}${y.partial ? " (год не закончился)" : ""}`}
              >
                {y === peak || y.partial ? (
                  <span className="text-[11px] text-muted-foreground">
                    {formatNumber(y.n)}
                  </span>
                ) : null}
                <span
                  className={cn(
                    "w-full max-w-7 rounded-t",
                    y.partial ? "bg-conf-low" : "bg-conf-mid"
                  )}
                  style={{
                    height: `${(y.n / max) * 100}%`,
                    minHeight: y.n ? 3 : 0,
                  }}
                />
              </div>
            ))}
          </div>
          <div
            className="grid gap-1 text-center text-[11px] text-muted-foreground"
            style={{
              gridTemplateColumns: `repeat(${years.length}, minmax(0, 1fr))`,
            }}
            aria-hidden="true"
          >
            {years.map((y, i) => (
              <span key={y.year}>
                {(years.length - 1 - i) % 2 === 0
                  ? `’${String(y.year).slice(2)}`
                  : ""}
              </span>
            ))}
          </div>
          <figcaption className="text-xs text-muted-foreground">
            Научные публикации по точному названию (OpenAlex), последние{" "}
            {CHART_YEARS} {plural(CHART_YEARS, "год", "года", "лет")}.
            Приглушённый столбик — текущий год, он ещё не закончился.
          </figcaption>
        </figure>
      ) : (
        <p className="text-sm text-muted-foreground">
          Статистика публикаций по годам не собрана: источник не ответил за
          отведённое время.
        </p>
      )}
      {facts.length > 0 ? (
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
          {facts.map(([label, value]) => (
            <div key={label} className="flex flex-col gap-0.5">
              <dt className="text-xs text-muted-foreground">{label}</dt>
              <dd className="font-medium">{value}</dd>
            </div>
          ))}
        </dl>
      ) : null}
    </div>
  )
}

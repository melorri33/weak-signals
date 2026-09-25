import { useLayoutEffect, useMemo, useRef, useState } from "react"
import { useNavigate } from "react-router"

import type { RunEvidence, SearchResult } from "@/api/client"
import { type MapPoint, mapPoints } from "@/lib/evidence"
import { confidenceLevel, formatNumber, percent, plural } from "@/lib/format"
import { cn } from "@/lib/utils"

const MARGIN = { top: 20, right: 20, bottom: 40, left: 60 }
const RADIUS = 7
const LABEL_MAX = 28
const LINE = 15
const CHAR_W = 6.6

const CONF_FILL = {
  high: "var(--conf-high)",
  mid: "var(--conf-mid)",
  low: "var(--conf-low)",
} as const

/**
 * Карта сигналов — классическая схема из теории слабых сигналов на измеренных числах.
 * По горизонтали — сколько всего публикаций (логарифм), по вертикали — какая их доля вышла за последние
 * три года. Слабый сигнал — слева вверху: публикаций мало, и почти все свежие. Зрелое — справа внизу.
 * Порогов на карте нет: отсев делают правила, карта только показывает, где оказался каждый кандидат.
 */
export function SignalMap({
  result,
  evidence,
  compact = false,
}: {
  result: SearchResult
  evidence: RunEvidence
  compact?: boolean
}) {
  const navigate = useNavigate()
  const [ref, width] = useWidth<HTMLDivElement>()
  const [active, setActive] = useState<string | null>(null)
  const { points, unplaced } = useMemo(
    () => mapPoints(result, evidence),
    [result, evidence]
  )

  const height = compact ? 260 : width < 640 ? 300 : 400
  const innerW = Math.max(0, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const maxPubs = Math.max(10, ...points.map((p) => p.pubs))
  const ticks = logTicks(maxPubs)
  const xMax = Math.log1p(ticks[ticks.length - 1])
  const x = (pubs: number) => (Math.log1p(pubs) / xMax) * innerW
  const y = (recent: number) => (1 - recent) * innerH

  // Сначала отсеянные и прочие, сверху — выдача: её точки не должны прятаться под серыми.
  const ordered = [...points].sort(
    (a, b) => statusOrder(a.status) - statusOrder(b.status)
  )
  const labels = placeLabels(
    points.filter((p) => p.status === "top"),
    x,
    y,
    innerW
  )
  const current = points.find((p) => p.id === active)

  function open(point: MapPoint) {
    if (point.status !== "top") return
    navigate(`/run/${result.run_id}/signal/${encodeURIComponent(point.id)}`)
  }

  return (
    <figure className="flex flex-col gap-3">
      <Legend points={points} pending={result.status === "running"} />
      <div ref={ref} className="relative w-full" style={{ height }}>
        {width > 0 ? (
          <svg
            width={width}
            height={height}
            role="group"
            aria-label="Карта кандидатов: публикаций всего и доля свежих"
            className="block overflow-visible"
          >
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <rect
                width={innerW * 0.45}
                height={innerH * 0.45}
                className="fill-accent/60"
                rx={8}
              />
              <text x={10} y={18} className="fill-muted-foreground text-[11px]">
                мало публикаций, почти все свежие
              </text>
              <text
                x={innerW - 8}
                y={innerH - 10}
                textAnchor="end"
                className="fill-muted-foreground text-[11px]"
              >
                много публикаций, давно
              </text>
              {[0, 0.5, 1].map((r) => (
                <g key={r} transform={`translate(0,${y(r)})`}>
                  <line x2={innerW} className="stroke-border" />
                  <text
                    x={-8}
                    dy="0.32em"
                    textAnchor="end"
                    className="fill-muted-foreground text-[11px]"
                  >
                    {percent(r)}
                  </text>
                </g>
              ))}
              {ticks.map((t) => (
                <g key={t} transform={`translate(${x(t)},0)`}>
                  <line y2={innerH} className="stroke-border/70" />
                  <text
                    y={innerH + 18}
                    textAnchor="middle"
                    className="fill-muted-foreground text-[11px]"
                  >
                    {formatNumber(t)}
                  </text>
                </g>
              ))}
              <text
                x={innerW}
                y={innerH + 34}
                textAnchor="end"
                className="fill-muted-foreground text-[11px]"
              >
                публикаций всего →
              </text>
              <text
                transform={`translate(${-48},${innerH / 2}) rotate(-90)`}
                textAnchor="middle"
                className="fill-muted-foreground text-[11px]"
              >
                доля публикаций за 3 года
              </text>
              {ordered.map((point) => (
                <Point
                  key={point.id}
                  point={point}
                  cx={x(point.pubs)}
                  cy={y(point.recent)}
                  active={point.id === active}
                  onActive={setActive}
                  onOpen={open}
                />
              ))}
              {labels.map((label) => (
                <text
                  key={label.id}
                  x={label.x}
                  y={label.y}
                  textAnchor={label.anchor}
                  className="pointer-events-none fill-foreground text-xs font-medium"
                  style={{
                    paintOrder: "stroke",
                    stroke: "var(--card)",
                    strokeWidth: 4,
                  }}
                >
                  {label.text}
                </text>
              ))}
            </g>
          </svg>
        ) : null}
        {current ? (
          <Tip
            point={current}
            left={MARGIN.left + x(current.pubs)}
            top={MARGIN.top + y(current.recent)}
            width={width}
          />
        ) : null}
      </div>
      <figcaption className="text-xs text-muted-foreground">
        Каждая точка — кандидат с измеренной статистикой публикаций. Слабые
        сигналы ищем в левом верхнем углу: публикаций мало, но почти все вышли
        за последние три года. Наведите на точку, чтобы увидеть числа; точка из
        выдачи открывает инсайт.
        {unplaced > 0
          ? ` Без статистики и не на карте: ${unplaced} ${plural(unplaced, "кандидат", "кандидата", "кандидатов")}.`
          : ""}
      </figcaption>
    </figure>
  )
}

function Point({
  point,
  cx,
  cy,
  active,
  onActive,
  onOpen,
}: {
  point: MapPoint
  cx: number
  cy: number
  active: boolean
  onActive: (id: string | null) => void
  onOpen: (point: MapPoint) => void
}) {
  const interactive = point.status === "top"
  const common = {
    tabIndex: 0,
    role: interactive ? "link" : "img",
    "aria-label": pointLabel(point),
    onMouseEnter: () => onActive(point.id),
    onMouseLeave: () => onActive(null),
    onFocus: () => onActive(point.id),
    onBlur: () => onActive(null),
    onClick: () => onOpen(point),
    onKeyDown: (event: React.KeyboardEvent) => {
      if (event.key === "Enter") onOpen(point)
    },
    className: cn(
      "outline-none focus-visible:[&>.ring]:stroke-ring",
      interactive && "cursor-pointer"
    ),
  }
  return (
    <g {...common} transform={`translate(${cx},${cy})`}>
      {/* Зона попадания больше самой точки. */}
      <circle r={RADIUS + 6} fill="transparent" />
      <circle
        className="ring"
        r={RADIUS + 3}
        fill="none"
        stroke={active ? "var(--ring)" : "transparent"}
        strokeWidth={2}
      />
      {point.status === "top" ? (
        <circle
          r={RADIUS}
          fill={CONF_FILL[confidenceLevel(point.score ?? 0)]}
          stroke="var(--card)"
          strokeWidth={2}
        />
      ) : point.status === "below" ? (
        <circle
          r={RADIUS - 1}
          fill="var(--card)"
          stroke="var(--conf-mid)"
          strokeWidth={2}
        />
      ) : point.status === "excluded" ? (
        <rect
          x={-(RADIUS - 2)}
          y={-(RADIUS - 2)}
          width={(RADIUS - 2) * 2}
          height={(RADIUS - 2) * 2}
          rx={2}
          fill="var(--muted-foreground)"
          opacity={0.55}
        />
      ) : (
        <circle
          r={RADIUS - 2}
          fill="var(--chart-2)"
          stroke="var(--card)"
          strokeWidth={2}
        />
      )}
    </g>
  )
}

function Legend({ points, pending }: { points: MapPoint[]; pending: boolean }) {
  const has = (status: MapPoint["status"]) =>
    points.some((p) => p.status === status)
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-muted-foreground">
      {has("top") ? (
        <li className="flex items-center gap-1.5">
          <span className="flex" aria-hidden="true">
            <span className="size-2.5 rounded-full bg-conf-low" />
            <span className="-ml-0.5 size-2.5 rounded-full bg-conf-mid" />
            <span className="-ml-0.5 size-2.5 rounded-full bg-conf-high" />
          </span>
          в выдаче, чем гуще — тем увереннее модель
        </li>
      ) : null}
      {has("below") ? (
        <li className="flex items-center gap-1.5">
          <span
            className="size-2.5 rounded-full border-2 border-conf-mid"
            aria-hidden="true"
          />
          оценён, ниже топ-15
        </li>
      ) : null}
      {has("excluded") ? (
        <li className="flex items-center gap-1.5">
          <span
            className="size-2.5 rounded-xs bg-muted-foreground/55"
            aria-hidden="true"
          />
          отсеян правилами
        </li>
      ) : null}
      {has("pending") ? (
        <li className="flex items-center gap-1.5">
          <span className="size-2 rounded-full bg-chart-2" aria-hidden="true" />
          {pending ? "ещё оценивается" : "не оценён"}
        </li>
      ) : null}
    </ul>
  )
}

function Tip({
  point,
  left,
  top,
  width,
}: {
  point: MapPoint
  left: number
  top: number
  width: number
}) {
  const flip = left > width - 240
  return (
    <div
      className="pointer-events-none absolute z-10 flex w-56 flex-col gap-1 rounded-lg border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-md"
      style={{
        left: flip ? left - 232 - RADIUS : left + RADIUS + 8,
        top: Math.max(0, top - 24),
      }}
      role="status"
    >
      <span className="text-sm font-medium">{point.name}</span>
      <span>
        {formatNumber(point.pubs)}{" "}
        {plural(point.pubs, "публикация", "публикации", "публикаций")}, за 3
        года — {percent(point.recent)}
      </span>
      <span className="text-muted-foreground">{statusText(point)}</span>
    </div>
  )
}

function statusText(point: MapPoint): string {
  switch (point.status) {
    case "top":
      return `В выдаче, уверенность ${percent(point.score ?? 0)}. Нажмите, чтобы открыть инсайт`
    case "below":
      return `Оценён на ${percent(point.score ?? 0)}, в выдачу не попал`
    case "excluded":
      return `Отсеян: ${point.reason?.toLowerCase() ?? "правило отсева"}`
    default:
      return "Ещё не оценён"
  }
}

function pointLabel(point: MapPoint): string {
  return `${point.name}: ${point.pubs} публикаций, за 3 года ${percent(point.recent)}. ${statusText(point)}`
}

function statusOrder(status: MapPoint["status"]): number {
  return { excluded: 0, pending: 1, below: 2, top: 3 }[status]
}

function logTicks(max: number): number[] {
  const ticks = [0, 1]
  for (let t = 10; ; t *= 10) {
    ticks.push(t)
    if (t >= max) break
  }
  return ticks
}

type Label = {
  id: string
  text: string
  x: number
  y: number
  anchor: "start" | "end"
}

/** Подписи только у точек выдачи; наезжающие сдвигаются вниз. Ширина — оценка по числу символов. */
function placeLabels(
  points: MapPoint[],
  x: (pubs: number) => number,
  y: (recent: number) => number,
  innerW: number
): Label[] {
  const placed: (Label & { w: number })[] = []
  const sorted = [...points].sort((a, b) => y(a.recent) - y(b.recent))
  for (const point of sorted) {
    const px = x(point.pubs)
    // Подпись идёт туда, где больше места, и укорачивается под него: на телефоне край близко.
    const roomRight = innerW + MARGIN.right - px - RADIUS - 8
    const roomLeft = px + MARGIN.left - RADIUS - 8
    const anchor: Label["anchor"] =
      roomRight >= Math.min(roomLeft, point.name.length * CHAR_W)
        ? "start"
        : "end"
    const fit = Math.floor((anchor === "start" ? roomRight : roomLeft) / CHAR_W)
    const limit = Math.max(6, Math.min(LABEL_MAX, fit))
    const text =
      point.name.length > limit
        ? `${point.name.slice(0, limit - 1)}…`
        : point.name
    const w = text.length * CHAR_W
    const lx = anchor === "start" ? px + RADIUS + 6 : px - RADIUS - 6
    let ly = y(point.recent) + 4
    const left = (l: { x: number; anchor: string; w: number }) =>
      l.anchor === "start" ? l.x : l.x - l.w
    while (
      placed.some(
        (p) =>
          Math.abs(p.y - ly) < LINE &&
          left(p) < left({ x: lx, anchor, w }) + w &&
          left({ x: lx, anchor, w }) < left(p) + p.w
      )
    ) {
      ly += LINE
    }
    placed.push({ id: point.id, text, x: lx, y: ly, anchor, w })
  }
  return placed
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    const node = ref.current
    if (!node) return
    setWidth(Math.round(node.getBoundingClientRect().width))
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.round(entry.contentRect.width))
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

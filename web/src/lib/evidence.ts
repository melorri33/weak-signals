/**
 * Признаки кандидатов (GET /search/{run_id}/evidence) → точки карты и ряды по годам.
 * Всё считается из измеренных чисел; нет числа — точки нет, а не «примерно ноль».
 */
import type { CandidateEvidence, RunEvidence, SearchResult } from "@/api/client"
import { REASON_LABEL, displayName } from "@/lib/format"

export type PointStatus = "top" | "below" | "excluded" | "pending"

export type MapPoint = {
  id: string
  name: string
  pubs: number
  recent: number
  status: PointStatus
  score?: number
  reason?: string
}

/** Сколько лет считаем «свежими»: те же три года, что у признака recent_share. */
const RECENT_YEARS = 3

export function yearSeries(
  byYear: Record<string, number> | null | undefined
): Map<number, number> {
  const series = new Map<number, number>()
  for (const [year, n] of Object.entries(byYear ?? {})) {
    const y = Number(year)
    if (Number.isFinite(y)) series.set(y, n)
  }
  return series
}

/** Доля публикаций за последние три года: признак модели, а если его нет — из ряда по годам. */
export function recentShare(item: CandidateEvidence): number | null {
  const fromModel = item.features.extra?.recent_share
  if (typeof fromModel === "number") return fromModel
  const series = yearSeries(item.pubs_by_year)
  const total = [...series.values()].reduce((a, b) => a + b, 0)
  if (total === 0) return null
  const since = new Date().getFullYear() - RECENT_YEARS + 1
  let recent = 0
  for (const [year, n] of series) if (year >= since) recent += n
  return recent / total
}

export function mapPoints(
  result: SearchResult,
  evidence: RunEvidence
): { points: MapPoint[]; unplaced: number } {
  const cards = new Map((result.top ?? []).map((c) => [c.candidate_id, c]))
  const scored = new Map((result.scored ?? []).map((c) => [c.candidate_id, c]))
  const excluded = new Map(
    (result.excluded ?? [])
      .filter((d) => d.excluded)
      .map((d) => [d.candidate_id, d])
  )
  const points: MapPoint[] = []
  let unplaced = 0
  for (const item of evidence.candidates ?? []) {
    const pubs = item.features.total_pubs
    const recent = recentShare(item)
    if (pubs === null || pubs === undefined || recent === null) {
      unplaced += 1
      continue
    }
    const card = cards.get(item.candidate_id)
    const score = scored.get(item.candidate_id)
    const cut = excluded.get(item.candidate_id)
    points.push({
      id: item.candidate_id,
      name: card
        ? displayName(card)
        : displayName({ name: item.name, name_ru: item.name_ru }),
      pubs,
      recent,
      status: card ? "top" : cut ? "excluded" : score ? "below" : "pending",
      score: card?.score ?? score?.score,
      reason: cut ? REASON_LABEL[cut.reason_code] : undefined,
    })
  }
  return { points, unplaced }
}

/** Карта сигналов строится только из измеренных чисел: нет числа — нет точки. */
import { describe, expect, it } from "vitest"

import type { CandidateEvidence, RunEvidence } from "@/api/client"
import { mapPoints, recentShare } from "@/lib/evidence"
import { exampleRun } from "./render"

function item(
  id: string,
  total: number | null,
  extra: Record<string, number | null> = {},
  pubs: Record<string, number> = {}
): CandidateEvidence {
  return {
    candidate_id: id,
    name: id,
    features: { candidate_id: id, total_pubs: total, extra },
    pubs_by_year: pubs,
  } as CandidateEvidence
}

describe("карта сигналов", () => {
  it("доля свежих — из признака модели, а без него из ряда по годам", () => {
    expect(recentShare(item("a", 10, { recent_share: 0.8 }))).toBe(0.8)
    const year = new Date().getFullYear()
    expect(
      recentShare(
        item("b", 10, {}, { [year]: 3, [year - 1]: 1, [year - 10]: 4 })
      )
    ).toBe(0.5)
    expect(recentShare(item("c", null))).toBeNull()
  })

  it("статус точки: выдача, ниже топ-15, отсеян; без статистики — не на карте", () => {
    const run = exampleRun()
    const top = run.top[0].candidate_id
    const cut = run.excluded.find((d) => d.excluded)!.candidate_id
    const below = run.scored.find(
      (s) => !run.top.some((c) => c.candidate_id === s.candidate_id)
    )!.candidate_id
    const evidence: RunEvidence = {
      run_id: run.run_id,
      candidates: [
        item(top, 12, { recent_share: 0.9 }),
        item(cut, 40000, { recent_share: 0.1 }),
        item(below, 30, { recent_share: 0.5 }),
        item("no-stats", null),
      ],
    }
    const { points, unplaced } = mapPoints(run, evidence)
    const status = Object.fromEntries(points.map((p) => [p.id, p.status]))
    expect(status).toEqual({
      [top]: "top",
      [cut]: "excluded",
      [below]: "below",
    })
    expect(unplaced).toBe(1)
    expect(points.find((p) => p.id === top)?.name).toBe("Пример технологии 1")
  })
})

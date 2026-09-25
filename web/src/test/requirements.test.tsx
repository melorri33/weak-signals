/**
 * Требования ТЗ к интерфейсу, проверяемые на фикстуре прогона.
 * Каждый тест называет пункт ТЗ, который он охраняет.
 */
import { fireEvent, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ApiError, activeRunId } from "@/api/client"
import { Funnel } from "@/components/funnel"
import {
  AllCandidates,
  ExcludedList,
  MethodPanel,
  TopSignals,
} from "@/components/run-tabs"
import { SourcesList } from "@/components/sources-list"
import { CONFIDENT_THRESHOLD, confidenceLevel, plural } from "@/lib/format"
import { InsightPage } from "@/pages/insight-page"
import { exampleRun, renderWithApp } from "./render"

afterEach(() => vi.unstubAllGlobals())

describe("источники", () => {
  it("у каждого источника: наименование, ссылка, дата, тип, язык оригинала, доверие", () => {
    const sources = exampleRun().top[0].sources
    renderWithApp(<SourcesList sources={sources} />)
    const items = screen.getAllByTestId("source")
    expect(items).toHaveLength(sources.length)
    const first = within(items[0])
    expect(
      first.getByRole("link", { name: /ПРИМЕР: название источника 1/ })
    ).toHaveAttribute("href", "https://example.org/doc/1")
    expect(first.getByText("01.11.2025")).toBeInTheDocument()
    expect(first.getByText("Научная статья")).toBeInTheDocument()
    expect(first.getByText("Английский")).toBeInTheDocument()
    expect(first.getByText("Высокое доверие")).toBeInTheDocument()
  })

  it("машинный перевод и сгенерированное резюме помечены рядом с источником", () => {
    renderWithApp(<SourcesList sources={exampleRun().top[0].sources} />)
    const first = within(screen.getAllByTestId("source")[0])
    expect(first.getByText("Машинный перевод")).toBeInTheDocument()
    expect(
      first.getByText("Резюме сгенерировано языковой моделью")
    ).toBeInTheDocument()
  })
})

describe("топ-15", () => {
  it("строка: название, уверенность, ключевые предикторы и ссылка на инсайт", () => {
    const run = exampleRun()
    renderWithApp(<TopSignals result={run} />)
    const rows = screen.getAllByRole("row").slice(1)
    expect(rows).toHaveLength(run.top.length)
    const first = within(rows[0])
    expect(first.getByText("Пример технологии 1")).toBeInTheDocument()
    expect(first.getByText("Высокая (91%)")).toBeInTheDocument()
    expect(first.getByText(/публикации растут/)).toBeInTheDocument()
    expect(
      first.getByRole("link", { name: "Смотреть инсайт" })
    ).toHaveAttribute("href", `/run/${run.run_id}/signal/example-tech-1`)
  })

  it("сигнал только на блогах и пресс-релизах помечен пониженной доверенностью", () => {
    renderWithApp(<TopSignals result={exampleRun()} />)
    const rows = screen.getAllByRole("row").slice(1)
    expect(
      within(rows[1]).getByText("Пониженная доверенность")
    ).toBeInTheDocument()
    expect(
      within(rows[0]).queryByText("Пониженная доверенность")
    ).not.toBeInTheDocument()
  })
})

describe("отсев", () => {
  it("исключённые сгруппированы по причине и показывают объяснение с цифрами", () => {
    const run = exampleRun()
    renderWithApp(<ExcludedList excluded={run.excluded} />)
    expect(
      screen.getByRole("heading", { name: "Зрелая технология" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Медийная тема" })
    ).toBeInTheDocument()
    for (const decision of run.excluded) {
      expect(screen.getByText(decision.reason_text)).toBeInTheDocument()
    }
  })

  it("все кандидаты: оценённые, ниже топ-15 и отсеянные — в одном списке", () => {
    renderWithApp(<AllCandidates result={exampleRun()} />)
    expect(screen.getAllByText("В топ-15")).toHaveLength(3)
    expect(screen.getByText("Ниже топ-15")).toBeInTheDocument()
    expect(screen.getByText("Отсеяно: зрелая технология")).toBeInTheDocument()
  })
})

describe("воронка", () => {
  it("показывает обработанные источники, кандидатов и уверенные сигналы и ведёт к спискам", () => {
    const run = exampleRun()
    const onSelect = vi.fn()
    renderWithApp(<Funnel result={run} onSelect={onSelect} />)
    expect(
      screen.getByRole("button", { name: /Обработано источников: 412/ })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /Уверенных сигналов: 2/ })
    ).toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: /Технологий-кандидатов: 48/ })
    )
    expect(onSelect).toHaveBeenCalledWith("candidates")
  })

  it("порог уверенности совпадает с бэкендом: строго больше 0.75", () => {
    expect(confidenceLevel(CONFIDENT_THRESHOLD)).toBe("mid")
    expect(confidenceLevel(0.76)).toBe("high")
    expect(confidenceLevel(0.49)).toBe("low")
  })
})

describe("как считали", () => {
  it("раскрывает модели по шагам и отказы источников", () => {
    renderWithApp(<MethodPanel result={exampleRun()} />)
    expect(
      screen.getByText("Поисковые фразы", { selector: "td" })
    ).toBeInTheDocument()
    expect(screen.getAllByText("qwen3:8b").length).toBeGreaterThan(0)
    expect(
      screen.getByText(/Не все источники ответили: 12 отказов/)
    ).toBeInTheDocument()
  })
})

describe("инсайт", () => {
  it("страница-отчёт: все разделы ТЗ и источники", async () => {
    const run = exampleRun()
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(run), { status: 200 }))
    )
    renderWithApp(<InsightPage />, {
      path: "/run/:runId/signal/:candidateId",
      route: `/run/${run.run_id}/signal/example-tech-1`,
    })
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
        "Пример технологии 1"
      )
    )
    for (const section of [
      "Описание технологии",
      "Потенциальное преимущество",
      "Кейс-пример",
      "Оценки в аналитических отчётах",
      "Почему это слабый сигнал",
      "Почему такая уверенность",
      "Источники",
    ]) {
      expect(
        screen.getByRole("heading", { level: 2, name: section })
      ).toBeInTheDocument()
    }
    expect(screen.getAllByTestId("source")).toHaveLength(
      run.top[0].sources.length
    )
  })

  it("пониженная доверенность предупреждает в начале отчёта", async () => {
    const run = exampleRun()
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(run), { status: 200 }))
    )
    renderWithApp(<InsightPage />, {
      path: "/run/:runId/signal/:candidateId",
      route: `/run/${run.run_id}/signal/${run.top[1].candidate_id}`,
    })
    expect(
      await screen.findByText("Пониженная доверенность")
    ).toBeInTheDocument()
  })
})

describe("мелочи, на которых легко ошибиться", () => {
  it("из 429 достаётся id идущего прогона — для ссылки «откройте идущий поиск»", () => {
    expect(
      activeRunId(
        new ApiError(
          429,
          "Уже идёт прогон 0523f15f6407 — дождитесь его окончания"
        )
      )
    ).toBe("0523f15f6407")
    expect(activeRunId(new ApiError(500, "прогон abcdef123456"))).toBeNull()
  })

  it("склонение по-русски", () => {
    expect(
      [1, 2, 5, 11, 21, 22].map((n) =>
        plural(n, "источник", "источника", "источников")
      )
    ).toEqual([
      "источник",
      "источника",
      "источников",
      "источников",
      "источник",
      "источника",
    ])
  })
})

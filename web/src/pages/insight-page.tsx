import { ArrowLeftIcon, PrinterIcon, TriangleAlertIcon } from "lucide-react"
import Markdown from "react-markdown"
import { Link, useParams } from "react-router"

import type { SignalCard, SourceRef } from "@/api/client"
import { ConfidenceBar, TrustBadge } from "@/components/levels"
import { ShapBars } from "@/components/shap-bars"
import { SourcesList } from "@/components/sources-list"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import {
  CONFIDENCE_LABEL,
  CONFIDENCE_TEXT,
  SOURCE_TYPE_LABEL,
  TOP_N,
  confidenceLevel,
  displayName,
  formatDate,
  hasOnlyLowTrust,
  percent,
  plural,
  analyticalSources,
} from "@/lib/format"
import { NotFoundPage } from "@/pages/not-found-page"
import { useRun } from "@/api/queries"

/** Инсайт — страница, похожая на документ-отчёт (ТЗ, раздел 7.2). Печатается в PDF как есть. */
export function InsightPage() {
  const { runId = "", candidateId = "" } = useParams()
  const { data: result, isPending, error } = useRun(runId)

  if (isPending) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-10 w-1/2" />
        <Skeleton className="h-96" />
      </div>
    )
  }
  if (error)
    return <NotFoundPage title="Поиск не найден" description={error.message} />

  const top = result.top ?? []
  const rank = top.findIndex((c) => c.candidate_id === candidateId)
  const card = top[rank]
  if (!card) {
    return (
      <NotFoundPage
        title="Сигнал не найден"
        description="В этом поиске нет такой технологии в топ-15."
      />
    )
  }

  return (
    <article className="flex flex-col gap-8">
      <div
        data-print="hide"
        className="flex items-center justify-between gap-4"
      >
        <Link
          to={`/run/${runId}`}
          className="inline-flex min-w-0 items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeftIcon className="size-4 shrink-0" aria-hidden="true" />
          <span className="truncate">К выдаче по запросу «{result.query}»</span>
        </Link>
        <Button variant="outline" onClick={() => window.print()}>
          <PrinterIcon data-icon="inline-start" />
          Печать или PDF
        </Button>
      </div>

      <header className="flex flex-col gap-3 border-b pb-8">
        <p className="text-sm text-muted-foreground">
          Слабый сигнал {rank + 1} из {top.length}, запрос «{result.query}»
        </p>
        <h1 className="text-3xl leading-tight font-semibold tracking-tight text-balance sm:text-4xl">
          {displayName(card)}
        </h1>
        {card.name_ru ? (
          <p className="text-lg text-muted-foreground" lang="en" translate="no">
            {card.name}
          </p>
        ) : null}
      </header>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-10 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="flex max-w-[44rem] min-w-0 flex-col gap-10">
          {hasOnlyLowTrust(card) ? (
            <Alert>
              <TriangleAlertIcon />
              <AlertTitle>Пониженная доверенность</AlertTitle>
              <AlertDescription>
                Все источники этого сигнала — блоги, соцсети, агрегаторы или
                пресс-релизы. По правилам ТЗ это только первичный индикатор:
                нужна проверка независимыми источниками.
              </AlertDescription>
            </Alert>
          ) : null}

          <ReportSection title="Описание технологии">
            <Prose text={card.description} />
          </ReportSection>
          <ReportSection title="Потенциальное преимущество">
            <Prose text={card.advantage} />
          </ReportSection>
          <ReportSection title="Кейс-пример">
            <Prose text={card.case_example} />
          </ReportSection>
          <ReportSection title="Оценки в аналитических отчётах">
            <AnalystViews sources={card.sources} />
          </ReportSection>
          <ReportSection title="Почему это слабый сигнал">
            <Prose text={card.why_weak_signal} />
          </ReportSection>
          <ReportSection
            title="Почему такая уверенность"
            lead={`Итоговая уверенность ${percent(card.score)}: ${CONFIDENCE_TEXT[confidenceLevel(card.score)].toLowerCase()}. Ниже — признаки, которые сильнее всего повлияли на оценку.`}
          >
            <ShapBars reasons={card.top_reasons ?? []} />
          </ReportSection>
          <ReportSection
            title="Источники"
            lead="Сигнал построен только по этим документам: ссылки взяты из собранных источников, языковая модель их не придумывает."
          >
            <SourcesList sources={card.sources} />
          </ReportSection>
          {card.report_md ? (
            <ReportSection title="Подробный отчёт">
              <div className="report-md font-serif leading-relaxed">
                <Markdown>{card.report_md}</Markdown>
              </div>
            </ReportSection>
          ) : null}
        </div>

        <SignalPassport card={card} rank={rank} total={top.length} />
      </div>
    </article>
  )
}

function ReportSection({
  title,
  lead,
  children,
}: {
  title: string
  lead?: string
  children: React.ReactNode
}) {
  return (
    <section className="flex break-inside-avoid-page flex-col gap-3">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      {lead ? <p className="text-sm text-muted-foreground">{lead}</p> : null}
      {children}
    </section>
  )
}

/** Текст карточки — в книжном шрифте: инсайт читается как аналитическая записка. */
function Prose({ text }: { text: string }) {
  if (!text.trim())
    return (
      <p className="text-sm text-muted-foreground">
        Модель не написала этот раздел.
      </p>
    )
  return (
    <div className="flex flex-col gap-3 font-serif text-[1.0625rem] leading-[1.7] text-pretty">
      {text.split(/\n{2,}/).map((paragraph, i) => (
        <p key={i}>{paragraph}</p>
      ))}
    </div>
  )
}

function AnalystViews({ sources }: { sources: SourceRef[] }) {
  const reports = analyticalSources(sources)
  if (reports.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Среди найденных источников нет аналитических отчётов и документов
        регуляторов. Оценка опирается на научные публикации и отраслевые СМИ из
        списка источников ниже. Для ранней технологии это ожидаемо: отчёты
        аналитиков появляются, когда рынок уже складывается.
      </p>
    )
  }
  return (
    <ul className="flex flex-col gap-4">
      {reports.map((source) => (
        <li
          key={source.document_id}
          className="flex flex-col gap-1.5 border-l-2 border-primary/40 pl-4"
        >
          <a
            href={source.url}
            target="_blank"
            rel="noreferrer noopener"
            className="font-medium hover:text-primary hover:underline"
          >
            {source.title}
          </a>
          <p className="text-xs text-muted-foreground">
            {SOURCE_TYPE_LABEL[source.source_type]},{" "}
            {formatDate(source.published)}
          </p>
          {source.ru_summary ? (
            <p className="font-serif leading-relaxed">{source.ru_summary}</p>
          ) : null}
        </li>
      ))}
    </ul>
  )
}

/** Паспорт сигнала: уверенность и состав источников — то, что жюри спросит первым. */
function SignalPassport({
  card,
  rank,
  total,
}: {
  card: SignalCard
  rank: number
  total: number
}) {
  const level = confidenceLevel(card.score)
  const trustCounts = (["high", "medium", "low"] as const)
    .map((trust) => ({
      trust,
      n: card.sources.filter((s) => s.trust === trust).length,
    }))
    .filter((c) => c.n > 0)
  return (
    <aside className="order-first h-fit lg:sticky lg:top-6 lg:order-none">
      <div className="flex flex-col gap-5 rounded-xl border bg-card p-5">
        <div className="flex flex-col gap-2">
          <p className="text-sm text-muted-foreground">Уверенность модели</p>
          <p className="text-4xl leading-none font-semibold tracking-tight">
            {percent(card.score)}
          </p>
          <ConfidenceBar score={card.score} />
          <p className="text-sm">
            <span className="font-medium">{CONFIDENCE_LABEL[level]}.</span>{" "}
            {CONFIDENCE_TEXT[level]}
          </p>
        </div>
        <Separator />
        <dl className="flex flex-col gap-3 text-sm">
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Место в выдаче</dt>
            <dd className="font-medium">
              {rank + 1} из {total} (топ-{TOP_N})
            </dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Источников</dt>
            <dd className="font-medium">
              {card.sources.length}{" "}
              {plural(
                card.sources.length,
                "документ",
                "документа",
                "документов"
              )}
            </dd>
          </div>
        </dl>
        <ul className="flex flex-col gap-2">
          {trustCounts.map(({ trust, n }) => (
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
    </aside>
  )
}

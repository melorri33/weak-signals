import { useQuery } from "@tanstack/react-query"
import { Link } from "react-router"

import { api, type RunSummary } from "@/api/client"
import { RunStatusBadge } from "@/components/run-status"
import { SearchForm } from "@/components/search-form"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty"
import { ConfidenceBar } from "@/components/levels"
import { Skeleton } from "@/components/ui/skeleton"
import {
  TOP_N,
  formatDateTime,
  formatNumber,
  percent,
  plural,
} from "@/lib/format"

export function HomePage() {
  return (
    <div className="flex flex-col gap-14">
      <div className="grid gap-10 pt-4 sm:pt-10 lg:grid-cols-[minmax(0,1fr)_20rem] lg:gap-16">
        <section className="flex max-w-3xl flex-col gap-6">
          <div className="flex flex-col gap-3">
            <h1 className="text-3xl leading-tight font-semibold tracking-tight text-balance sm:text-4xl">
              Какие технологии только зарождаются?
            </h1>
            <p className="max-w-2xl text-base text-pretty text-muted-foreground sm:text-lg">
              Введите направление. Сервис соберёт свежие статьи, препринты и
              техноновости, отсеет зрелое, хайп и шум и покажет {TOP_N}{" "}
              технологий на ранней стадии — с оценкой модели, объяснением и
              ссылками на документы.
            </p>
          </div>
          {/* Автофокус только с мышью: на телефоне он сразу открывает клавиатуру и прячет текст. */}
          <SearchForm
            autoFocus={window.matchMedia("(pointer: fine)").matches}
          />
          <p className="text-sm text-muted-foreground">
            Поиск идёт в реальном времени по открытым источникам и занимает
            несколько минут: на ноутбуке с видеокартой около шести.
          </p>
        </section>
        <HowItWorks />
      </div>
      <RecentRuns />
    </div>
  )
}

const METHOD_STEPS = [
  {
    title: "Собираем",
    text: "Техноновости, препринты arXiv и публикации OpenAlex по фразам, которые модель выводит из запроса",
  },
  {
    title: "Выделяем технологии",
    text: "Языковая модель выписывает из документов узкие названия, а не общие слова",
  },
  {
    title: "Отсеиваем",
    text: "Зрелое — тысячи публикаций и стандарты; хайп — много СМИ на одну работу; шум — только блоги и пресс-релизы",
  },
  {
    title: "Оцениваем и объясняем",
    text: "Классификатор по динамике публикаций; вклад каждого признака показан в карточке",
  },
]

/** Метод в четырёх шагах — жюри оценивает обоснованность, и она должна быть видна до первого клика. */
function HowItWorks() {
  return (
    <section
      aria-labelledby="how-it-works"
      className="flex flex-col gap-4 lg:border-l lg:pl-8"
    >
      <h2 id="how-it-works" className="text-base font-semibold">
        Как отделяем сигнал от шума
      </h2>
      <ol className="flex flex-col gap-4">
        {METHOD_STEPS.map((step, i) => (
          <li key={step.title} className="grid grid-cols-[1.5rem_1fr] gap-x-2">
            <span className="text-sm font-semibold text-primary">{i + 1}</span>
            <span className="text-sm font-medium">{step.title}</span>
            <span className="col-start-2 text-sm text-pretty text-muted-foreground">
              {step.text}
            </span>
          </li>
        ))}
      </ol>
    </section>
  )
}

function RecentRuns() {
  const { data, isPending, isError } = useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
  })

  return (
    <section className="flex flex-col gap-4" aria-labelledby="recent-runs">
      <h2 id="recent-runs" className="text-lg font-semibold">
        Последние поиски
      </h2>
      {isPending ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      ) : isError ? (
        <p className="text-sm text-muted-foreground">
          Список поисков недоступен: сервер не отвечает.
        </p>
      ) : data.length === 0 ? (
        <Empty className="border">
          <EmptyHeader>
            <EmptyTitle>Поисков пока не было</EmptyTitle>
            <EmptyDescription>
              Запустите первый — результат сохранится здесь и откроется
              мгновенно.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <RunCards runs={data} />
      )}
    </section>
  )
}

/** Поиски карточками: по каждой сразу видно, что нашлось, — топ-3 с уверенностью. */
function RunCards({ runs }: { runs: RunSummary[] }) {
  return (
    <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {runs.map((run) => (
        <li key={run.run_id}>
          <Link
            to={`/run/${run.run_id}`}
            className="group flex h-full flex-col gap-4 rounded-xl border bg-card p-5 transition-colors hover:border-primary/40 focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
          >
            <div className="flex flex-col gap-1">
              <h3 className="font-semibold text-balance group-hover:text-primary">
                {run.query}
              </h3>
              <p className="text-xs text-muted-foreground">
                {formatDateTime(run.started_at)}
                {run.documents_processed
                  ? `, ${formatNumber(run.documents_processed)} ${plural(run.documents_processed, "документ", "документа", "документов")}`
                  : ""}
              </p>
            </div>
            {run.status === "done" && (run.leaders ?? []).length > 0 ? (
              <ol className="flex flex-col gap-2.5">
                {(run.leaders ?? []).map((leader) => (
                  <li key={leader.name} className="flex flex-col gap-1">
                    <span className="flex items-baseline justify-between gap-3 text-sm">
                      <span className="truncate">{leader.name}</span>
                      <span className="shrink-0 font-medium">
                        {percent(leader.score)}
                      </span>
                    </span>
                    <ConfidenceBar score={leader.score} className="h-1" />
                  </li>
                ))}
              </ol>
            ) : (
              <RunStatusBadge status={run.status} stage={run.stage} />
            )}
            {run.status === "done" ? (
              <p className="mt-auto text-sm text-muted-foreground">
                {run.signals}{" "}
                {plural(run.signals, "сигнал", "сигнала", "сигналов")},
                уверенных {run.confident_signals}
              </p>
            ) : null}
          </Link>
        </li>
      ))}
    </ul>
  )
}

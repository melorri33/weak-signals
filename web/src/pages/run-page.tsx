import { ArrowLeftIcon, DownloadIcon, RotateCcwIcon } from "lucide-react"
import { Link, useParams, useSearchParams } from "react-router"

import { ApiError, type SearchResult } from "@/api/client"
import { useEvidence, useRun, useStartSearch } from "@/api/queries"
import { Funnel, type RunTab } from "@/components/funnel"
import {
  AllCandidates,
  ExcludedList,
  MethodPanel,
  SourceFailures,
  SourceFailuresNote,
  TopSignals,
} from "@/components/run-tabs"
import { SignalMap } from "@/components/signal-map"
import { StageProgress } from "@/components/stage-progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TOP_N, formatDateTime, formatDuration, plural } from "@/lib/format"
import { NotFoundPage } from "@/pages/not-found-page"

const TABS: RunTab[] = ["top", "excluded", "candidates", "method"]

export function RunPage() {
  const { runId = "" } = useParams()
  const { data: result, error, isPending } = useRun(runId)

  if (isPending) return <RunSkeleton />
  if (error) {
    if (error instanceof ApiError && error.status === 404) {
      return (
        <NotFoundPage
          title="Поиск не найден"
          description="Возможно, сервер перезапускали, а база не подключена."
        />
      )
    }
    return (
      <Alert variant="destructive">
        <AlertTitle>Не удалось загрузить поиск</AlertTitle>
        <AlertDescription>{error.message}</AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="flex flex-col gap-8">
      <RunHeader result={result} />
      {result.status === "running" ? <LiveRun result={result} /> : null}
      {result.status === "error" ? <RunError result={result} /> : null}
      {result.status === "done" ? <RunResult result={result} /> : null}
    </div>
  )
}

/** Идущий прогон: шаги, а как только посчитаны признаки — кандидаты на карте. */
function LiveRun({ result }: { result: SearchResult }) {
  const { data: evidence } = useEvidence(result.run_id, true)
  return (
    <div className="flex flex-col gap-8">
      <StageProgress result={result} />
      {evidence ? (
        <section
          aria-labelledby="live-map"
          className="flex flex-col gap-3 rounded-xl border bg-card p-4 sm:p-6"
        >
          <h2 id="live-map" className="text-lg font-semibold">
            Кандидаты на карте
          </h2>
          <SignalMap result={result} evidence={evidence} />
        </section>
      ) : null}
    </div>
  )
}

function RunHeader({ result }: { result: SearchResult }) {
  return (
    <header className="flex flex-col gap-4">
      <Link
        to="/"
        className="inline-flex w-fit items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeftIcon className="size-4" aria-hidden="true" />
        Новый поиск
      </Link>
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="flex min-w-0 flex-col gap-1.5">
          <p className="text-sm text-muted-foreground">Поисковый запрос</p>
          <h1 className="text-2xl leading-tight font-semibold tracking-tight text-balance break-words sm:text-3xl">
            {result.query}
          </h1>
          <p className="text-sm text-muted-foreground">
            {result.status === "running"
              ? `Идёт поиск: ${result.stage || "в очереди"}`
              : `${formatDateTime(result.started_at)}${result.duration_s ? `, поиск занял ${formatDuration(result.duration_s)}` : ""}`}
          </p>
        </div>
        {result.status === "done" ? (
          <Button
            variant="outline"
            onClick={() => downloadJson(result)}
            data-present="hide"
          >
            <DownloadIcon data-icon="inline-start" />
            Скачать JSON
          </Button>
        ) : null}
      </div>
    </header>
  )
}

function RunResult({ result }: { result: SearchResult }) {
  const { data: evidence } = useEvidence(result.run_id, false)
  const [params, setParams] = useSearchParams()
  const requested = params.get("tab") as RunTab | null
  const tab: RunTab = requested && TABS.includes(requested) ? requested : "top"

  function select(next: RunTab) {
    setParams(next === "top" ? {} : { tab: next }, {
      replace: true,
      preventScrollReset: true,
    })
  }

  function open(next: RunTab) {
    select(next)
    document
      .getElementById("run-tabs")
      ?.scrollIntoView({ behavior: "smooth", block: "start" })
  }

  return (
    <div className="flex flex-col gap-6">
      <Verdict result={result} />
      <Funnel result={result} onSelect={open} />
      <div data-present="hide">
        <SourceFailuresNote
          failures={result.source_failures ?? []}
          onDetails={() => open("method")}
        />
      </div>
      {evidence ? (
        <section
          aria-labelledby="signal-map"
          className="flex flex-col gap-3 rounded-xl border bg-card p-4 sm:p-6"
        >
          <h2 id="signal-map" className="text-lg font-semibold">
            Карта сигналов
          </h2>
          <SignalMap result={result} evidence={evidence} />
        </section>
      ) : null}
      <Tabs
        id="run-tabs"
        value={tab}
        onValueChange={(value) => select(value as RunTab)}
        className="mt-2 scroll-mt-6 gap-5"
      >
        <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
          <TabsList variant="line" className="h-10">
            <TabsTrigger value="top">Топ-15 сигналов</TabsTrigger>
            <TabsTrigger value="excluded">
              Отсеяно ({result.excluded?.filter((d) => d.excluded).length ?? 0})
            </TabsTrigger>
            <TabsTrigger value="candidates">
              Все кандидаты (
              {(result.scored?.length ?? 0) +
                (result.excluded?.filter((d) => d.excluded).length ?? 0)}
              )
            </TabsTrigger>
            <TabsTrigger value="method">Как считали</TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="top">
          <TopSignals result={result} evidence={evidence ?? undefined} />
        </TabsContent>
        <TabsContent value="excluded">
          <ExcludedList excluded={result.excluded ?? []} />
        </TabsContent>
        <TabsContent value="candidates">
          <AllCandidates result={result} />
        </TabsContent>
        <TabsContent value="method">
          <MethodPanel result={result} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

/**
 * Итог одной фразой. Если в выдаче меньше 15, говорим почему — иначе неполный топ выглядит как сбой.
 */
function Verdict({ result }: { result: SearchResult }) {
  const top = result.top?.length ?? 0
  const confident = result.confident_signals ?? 0
  if (top === 0) return null
  return (
    <p className="max-w-3xl text-lg text-pretty">
      {top} {plural(top, "сигнал", "сигнала", "сигналов")} в выдаче
      {confident > 0
        ? `, из них ${confident} ${plural(confident, "уверенный", "уверенных", "уверенных")}.`
        : ", уверенных среди них нет."}
      {top < TOP_N ? (
        <span className="text-muted-foreground">
          {" "}
          Топ-{TOP_N} заполнен не целиком: подтверждающие документы нашлись не
          для всех кандидатов, а без документов сигнал в выдачу не попадает.
        </span>
      ) : null}
    </p>
  )
}

function RunError({ result }: { result: SearchResult }) {
  const retry = useStartSearch()
  return (
    <div className="flex flex-col gap-4">
      <Alert variant="destructive">
        <AlertTitle>Поиск остановился</AlertTitle>
        <AlertDescription>
          {result.error ?? "Причина не записана — смотрите лог сервера."}
        </AlertDescription>
      </Alert>
      <SourceFailures failures={result.source_failures ?? []} />
      <div>
        <Button
          onClick={() => retry.mutate(result.query)}
          disabled={retry.isPending}
        >
          <RotateCcwIcon data-icon="inline-start" />
          Повторить поиск
        </Button>
        {retry.error ? (
          <p className="mt-2 text-sm text-destructive">{retry.error.message}</p>
        ) : null}
      </div>
    </div>
  )
}

function RunSkeleton() {
  return (
    <div
      className="flex flex-col gap-6"
      aria-busy="true"
      aria-label="Загрузка…"
    >
      <Skeleton className="h-9 w-2/3" />
      <Skeleton className="h-32" />
      <Skeleton className="h-80" />
    </div>
  )
}

function downloadJson(result: SearchResult) {
  const blob = new Blob([JSON.stringify(result, null, 2)], {
    type: "application/json",
  })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = `${result.run_id}.json`
  link.click()
  URL.revokeObjectURL(url)
}

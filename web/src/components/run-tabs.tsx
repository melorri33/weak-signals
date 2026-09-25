import { useMemo, useState } from "react"
import {
  ArrowDownIcon,
  ArrowUpIcon,
  SearchIcon,
  TriangleAlertIcon,
} from "lucide-react"
import { Link } from "react-router"

import type {
  FilterDecision,
  ModelCall,
  ReasonCode,
  SearchResult,
  SignalCard,
  SourceFailure,
} from "@/api/client"
import { ConfidenceBadge, ConfidenceBar, TrustDots } from "@/components/levels"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  REASON_LABEL,
  REASON_MEANING,
  REASON_ORDER,
  displayName,
  formatDuration,
  formatNumber,
  hasOnlyLowTrust,
  percent,
  plural,
} from "@/lib/format"
import { cn } from "@/lib/utils"

const KEY_PREDICTORS = 2

/** Топ-15 — таблица по образцу интерфейса из ТЗ: технология, скоринг, ключевые предикторы, инсайт. */
export function TopSignals({ result }: { result: SearchResult }) {
  const top = result.top ?? []
  if (top.length === 0) {
    return (
      <Empty className="border">
        <EmptyHeader>
          <EmptyTitle>Сигналов не нашлось</EmptyTitle>
          <EmptyDescription>
            Все кандидаты отсеяны или у них не оказалось подтверждённых
            источников. Посмотрите вкладку «Отсеяно».
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }
  return (
    <div className="overflow-x-auto rounded-xl border bg-card">
      <Table className="min-w-[56rem]">
        <TableHeader>
          <TableRow>
            <TableHead className="w-10 pl-4 text-right">№</TableHead>
            <TableHead>Технология</TableHead>
            <TableHead className="w-44">Уверенность модели</TableHead>
            <TableHead>Ключевые предикторы</TableHead>
            <TableHead className="w-28">Источники</TableHead>
            <TableHead className="w-36 pr-4">
              <span className="sr-only">Инсайт</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {top.map((card, i) => (
            <SignalRow
              key={card.candidate_id}
              card={card}
              rank={i + 1}
              runId={result.run_id}
            />
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function SignalRow({
  card,
  rank,
  runId,
}: {
  card: SignalCard
  rank: number
  runId: string
}) {
  const reasons = [...(card.top_reasons ?? [])]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, KEY_PREDICTORS)
  const href = `/run/${runId}/signal/${encodeURIComponent(card.candidate_id)}`
  return (
    <TableRow className="relative align-top">
      <TableCell className="pt-4 pl-4 text-right text-muted-foreground">
        {rank}
      </TableCell>
      <TableCell className="max-w-72 py-3 whitespace-normal">
        <div className="flex flex-col gap-0.5">
          <span className="font-medium">{displayName(card)}</span>
          {card.name_ru ? (
            <span
              className="text-xs text-muted-foreground"
              translate="no"
              lang="en"
            >
              {card.name}
            </span>
          ) : null}
          {hasOnlyLowTrust(card) ? (
            <span className="mt-1 inline-flex items-center gap-1 text-xs text-level-low">
              <TriangleAlertIcon className="size-3.5" aria-hidden="true" />
              Пониженная доверенность
            </span>
          ) : null}
        </div>
      </TableCell>
      <TableCell className="py-3">
        <div className="flex flex-col gap-2">
          <ConfidenceBadge score={card.score} />
          <ConfidenceBar score={card.score} className="max-w-36" />
        </div>
      </TableCell>
      <TableCell className="py-3 whitespace-normal">
        <ul className="flex flex-col gap-1 text-sm">
          {reasons.map((reason) => (
            <li key={reason.feature} className="flex gap-1.5">
              <ReasonArrow contribution={reason.contribution} />
              <span>{reason.text}</span>
            </li>
          ))}
          {reasons.length === 0 ? (
            <li className="text-muted-foreground">
              Объяснение модели недоступно
            </li>
          ) : null}
        </ul>
      </TableCell>
      <TableCell className="py-3">
        <div className="flex flex-col gap-1.5">
          <span className="text-sm">
            {card.sources.length}{" "}
            {plural(card.sources.length, "источник", "источника", "источников")}
          </span>
          <TrustDots trusts={card.sources.map((s) => s.trust)} />
        </div>
      </TableCell>
      <TableCell className="py-3 pr-4">
        <Link
          to={href}
          className="font-medium text-primary underline-offset-4 after:absolute after:inset-0 hover:underline focus-visible:outline-none after:focus-visible:ring-3 after:focus-visible:ring-ring/50"
        >
          Смотреть инсайт
        </Link>
      </TableCell>
    </TableRow>
  )
}

export function ReasonArrow({ contribution }: { contribution: number }) {
  return contribution >= 0 ? (
    <ArrowUpIcon
      className="mt-0.5 size-4 shrink-0 text-shap-for"
      aria-label="за слабый сигнал"
    />
  ) : (
    <ArrowDownIcon
      className="mt-0.5 size-4 shrink-0 text-shap-against"
      aria-label="против"
    />
  )
}

/** Отсеянное с причиной — ТЗ требует показать логику исключения зрелого, хайпа и шума. */
export function ExcludedList({ excluded }: { excluded: FilterDecision[] }) {
  const groups = useMemo(() => {
    const byCode = new Map<ReasonCode, FilterDecision[]>()
    for (const decision of excluded.filter((d) => d.excluded)) {
      byCode.set(decision.reason_code, [
        ...(byCode.get(decision.reason_code) ?? []),
        decision,
      ])
    }
    return REASON_ORDER.filter((code) => byCode.has(code)).map((code) => ({
      code,
      items: byCode.get(code)!,
    }))
  }, [excluded])

  if (groups.length === 0) {
    return (
      <Empty className="border">
        <EmptyHeader>
          <EmptyTitle>Правила никого не отсеяли</EmptyTitle>
          <EmptyDescription>
            Все кандидаты прошли к оценке модели: зрелого, хайпа и шума среди
            них не нашлось.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }
  return (
    <div className="flex flex-col gap-6">
      <p className="max-w-3xl text-sm text-muted-foreground">
        Правила срабатывают только на измеренных данных и всегда называют цифры.
        Нет данных — правило не срабатывает: отсутствие статистики не считается
        признаком зрелости.
      </p>
      {groups.map(({ code, items }) => (
        <section
          key={code}
          className="flex flex-col gap-3"
          aria-labelledby={`reason-${code}`}
        >
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h2 id={`reason-${code}`} className="font-semibold">
              {REASON_LABEL[code]}
            </h2>
            <span className="text-sm text-muted-foreground">
              {items.length}{" "}
              {plural(items.length, "технология", "технологии", "технологий")}
            </span>
          </div>
          <p className="-mt-2 text-sm text-muted-foreground">
            {REASON_MEANING[code]}
          </p>
          <ul className="divide-y rounded-xl border bg-card">
            {items.map((item) => (
              <li
                key={item.candidate_id}
                className="flex flex-col gap-1 px-4 py-3 sm:flex-row sm:gap-6"
              >
                <span className="font-medium sm:w-64 sm:shrink-0">
                  {item.name}
                </span>
                <span className="text-sm text-muted-foreground">
                  {item.reason_text}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}

type CandidateRow = {
  id: string
  name: string
  score: number | null
  status: "top" | "below" | "excluded"
  reason?: string
}

/** Все кандидаты: и оценённые моделью, и отсеянные правилами, — чтобы было видно, откуда взялся топ-15. */
export function AllCandidates({ result }: { result: SearchResult }) {
  const [filter, setFilter] = useState("")
  const rows = useMemo<CandidateRow[]>(() => {
    const inTop = new Set((result.top ?? []).map((c) => c.candidate_id))
    const scored: CandidateRow[] = (result.scored ?? []).map((c) => ({
      id: c.candidate_id,
      name: c.name,
      score: c.score,
      status: inTop.has(c.candidate_id) ? "top" : "below",
    }))
    const excluded: CandidateRow[] = (result.excluded ?? [])
      .filter((d) => d.excluded)
      .map((d) => ({
        id: d.candidate_id,
        name: d.name,
        score: null,
        status: "excluded",
        reason: REASON_LABEL[d.reason_code],
      }))
    return [...scored, ...excluded]
  }, [result])

  const needle = filter.trim().toLowerCase()
  const visible = needle
    ? rows.filter((r) => r.name.toLowerCase().includes(needle))
    : rows

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-muted-foreground">
          Сначала оценённые моделью по убыванию уверенности, затем отсеянные
          правилами. В топ-15 попадают первые кандидаты, для которых нашлись
          подтверждающие документы; варианты одного названия склеиваются.
        </p>
        <InputGroup className="sm:w-72">
          <InputGroupAddon>
            <SearchIcon />
          </InputGroupAddon>
          <InputGroupInput
            aria-label="Найти кандидата"
            placeholder="Найти кандидата…"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          />
        </InputGroup>
      </div>
      <div className="overflow-x-auto rounded-xl border bg-card">
        <Table className="min-w-[36rem]">
          <TableHeader>
            <TableRow>
              <TableHead className="pl-4">Технология</TableHead>
              <TableHead className="w-48">Оценка модели</TableHead>
              <TableHead className="w-52 pr-4">Итог</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.map((row) => (
              <TableRow key={`${row.status}-${row.id}`}>
                <TableCell className="pl-4 font-medium whitespace-normal">
                  {row.status === "top" ? (
                    <Link
                      to={`/run/${result.run_id}/signal/${encodeURIComponent(row.id)}`}
                      className="text-primary underline-offset-4 hover:underline"
                    >
                      {row.name}
                    </Link>
                  ) : (
                    row.name
                  )}
                </TableCell>
                <TableCell>
                  {row.score === null ? (
                    <span className="text-muted-foreground">
                      не оценивалась
                    </span>
                  ) : (
                    <div className="flex items-center gap-3">
                      <span className="w-10 text-right">
                        {percent(row.score)}
                      </span>
                      <ConfidenceBar score={row.score} className="max-w-28" />
                    </div>
                  )}
                </TableCell>
                <TableCell className="pr-4">
                  {row.status === "top" ? (
                    <Badge variant="default">В топ-15</Badge>
                  ) : row.status === "below" ? (
                    <Badge variant="outline">Ниже топ-15</Badge>
                  ) : (
                    <Badge variant="low">
                      Отсеяно: {row.reason?.toLowerCase()}
                    </Badge>
                  )}
                </TableCell>
              </TableRow>
            ))}
            {visible.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={3}
                  className="py-8 text-center text-muted-foreground"
                >
                  Ничего не нашлось по «{filter}»
                </TableCell>
              </TableRow>
            ) : null}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}

const STEP_LABEL: Record<string, string> = {
  expand_query: "Поисковые фразы",
  expand_query_en: "Добор английских фраз",
  extract_candidates: "Выделение технологий",
  make_card: "Карточки сигналов",
  score: "Оценка сигнала (CatBoost + SHAP)",
  name_model: "Модель названий",
  embeddings: "Эмбеддинги",
  translate: "Перевод",
}

/** «Как считали»: фразы, журнал моделей и отказы источников — ТЗ требует раскрывать модели и логировать вызовы. */
export function MethodPanel({ result }: { result: SearchResult }) {
  const byStep = useMemo(
    () => summarizeCalls(result.model_calls ?? []),
    [result.model_calls]
  )
  return (
    <div className="flex flex-col gap-8">
      <SourceFailures failures={result.source_failures ?? []} />
      <section className="flex flex-col gap-3" aria-labelledby="phrases">
        <h2 id="phrases" className="font-semibold">
          Поисковые фразы
        </h2>
        <p className="text-sm text-muted-foreground">
          Языковая модель переформулировала запрос в фразы, по которым шёл
          поиск. Список технологий заранее не задан.
        </p>
        <ul className="flex flex-wrap gap-2">
          {(result.expanded_phrases ?? []).map((phrase) => (
            <li
              key={phrase}
              className="rounded-md border bg-card px-2.5 py-1 text-sm"
              translate="no"
            >
              {phrase}
            </li>
          ))}
        </ul>
      </section>
      <section className="flex flex-col gap-3" aria-labelledby="models">
        <h2 id="models" className="font-semibold">
          Какие модели работали
        </h2>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Решение «сигнал или нет» принимает не языковая модель, а классификатор
          по измеримым признакам. Языковая модель только формулирует фразы,
          выписывает названия технологий и пишет карточки по найденным
          документам. Каждый вызов записан в журнал.
        </p>
        <div className="overflow-x-auto rounded-xl border bg-card">
          <Table className="min-w-[36rem]">
            <TableHeader>
              <TableRow>
                <TableHead className="pl-4">Шаг</TableHead>
                <TableHead>Модель</TableHead>
                <TableHead>Где работает</TableHead>
                <TableHead className="text-right">Вызовов</TableHead>
                <TableHead className="pr-4 text-right">Время</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {byStep.map((row) => (
                <TableRow key={`${row.step}-${row.model}`}>
                  <TableCell className="pl-4 font-medium">
                    {STEP_LABEL[row.step] ?? row.step}
                  </TableCell>
                  <TableCell translate="no">{row.model}</TableCell>
                  <TableCell className="text-muted-foreground">
                    {row.provider}
                  </TableCell>
                  <TableCell className="text-right">{row.count}</TableCell>
                  <TableCell className="pr-4 text-right">
                    {formatDuration(row.ms / 1000)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </section>
      <dl className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
        <Fact
          label="Документов обработано"
          value={formatNumber(result.documents_processed ?? 0)}
        />
        <Fact
          label="Кандидатов"
          value={formatNumber(result.candidates_found ?? 0)}
        />
        <Fact
          label="Длительность"
          value={result.duration_s ? formatDuration(result.duration_s) : "—"}
        />
        <Fact label="Идентификатор" value={result.run_id} />
      </dl>
    </div>
  )
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium break-all">{value}</dd>
    </div>
  )
}

function summarizeCalls(calls: ModelCall[]) {
  const rows = new Map<
    string,
    { step: string; model: string; provider: string; count: number; ms: number }
  >()
  for (const call of calls) {
    const key = `${call.step}|${call.model}|${call.provider}`
    const row = rows.get(key) ?? {
      step: call.step,
      model: call.model,
      provider: call.provider,
      count: 0,
      ms: 0,
    }
    row.count += 1
    row.ms += call.duration_ms
    rows.set(key, row)
  }
  return [...rows.values()]
}

/** Отказ источника не валит прогон, поэтому его надо показать: иначе выдача выглядит полной. */
export function SourceFailures({
  failures,
  className,
}: {
  failures: SourceFailure[]
  className?: string
}) {
  if (failures.length === 0) return null
  const total = failures.reduce((sum, f) => sum + (f.count ?? 1), 0)
  return (
    <Alert className={cn(className)}>
      <TriangleAlertIcon />
      <AlertTitle>
        Не все источники ответили: {total}{" "}
        {plural(total, "отказ", "отказа", "отказов")}
      </AlertTitle>
      <AlertDescription>
        <p>
          Поиск продолжился без них. Признаки части кандидатов посчитаны по
          неполным данным.
        </p>
        <ul className="mt-1 flex flex-col gap-0.5">
          {failures.map((f) => (
            <li key={f.source}>
              <span className="font-medium text-foreground">{f.source}</span>
              {f.count && f.count > 1 ? `, ${f.count} раз` : ""}
              {f.detail ? `: ${f.detail}` : ""}
            </li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  )
}

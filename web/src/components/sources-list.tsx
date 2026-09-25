import { ExternalLinkIcon, SparklesIcon } from "lucide-react"

import type { SourceRef } from "@/api/client"
import { TrustBadge } from "@/components/levels"
import { Badge } from "@/components/ui/badge"
import { SOURCE_TYPE_LABEL, formatDate, languageLabel } from "@/lib/format"

/**
 * Источники сигнала. По ТЗ у каждого: наименование, ссылка, дата публикации, тип, язык оригинала,
 * уровень доверия. Для зарубежных — русское резюме с пометкой, если оно машинное.
 */
export function SourcesList({ sources }: { sources: SourceRef[] }) {
  return (
    <ol className="flex flex-col divide-y rounded-xl border bg-card">
      {sources.map((source, i) => (
        <li
          key={source.document_id}
          className="flex flex-col gap-3 p-4 sm:p-5"
          data-testid="source"
        >
          <div className="flex gap-3">
            <span className="w-5 shrink-0 pt-0.5 text-right text-sm text-muted-foreground">
              {i + 1}
            </span>
            <div className="flex min-w-0 flex-col gap-3">
              <a
                href={source.url}
                target="_blank"
                rel="noreferrer noopener"
                className="group inline-flex items-start gap-1.5 font-medium break-words text-foreground underline-offset-4 hover:text-primary hover:underline"
                lang={source.language}
              >
                <span>{source.title}</span>
                <ExternalLinkIcon
                  className="mt-1 size-3.5 shrink-0 text-muted-foreground group-hover:text-primary"
                  aria-hidden="true"
                />
                <span className="sr-only">(откроется в новой вкладке)</span>
              </a>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
                <Attr label="Дата публикации">
                  {formatDate(source.published)}
                </Attr>
                <Attr label="Тип источника">
                  {SOURCE_TYPE_LABEL[source.source_type]}
                </Attr>
                <Attr label="Язык оригинала">
                  {languageLabel(source.language)}
                </Attr>
                <Attr label="Доверие">
                  <TrustBadge trust={source.trust} />
                </Attr>
              </dl>
              <p
                className="truncate text-xs text-muted-foreground"
                title={source.url}
              >
                {source.url}
              </p>
              {source.ru_summary ? <Summary source={source} /> : null}
            </div>
          </div>
        </li>
      ))}
    </ol>
  )
}

function Attr({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd>{children}</dd>
    </div>
  )
}

function Summary({ source }: { source: SourceRef }) {
  const marks = [
    source.is_machine_translated ? "Машинный перевод" : null,
    source.summary_is_generated
      ? "Резюме сгенерировано языковой моделью"
      : null,
  ].filter(Boolean) as string[]
  return (
    <div className="flex flex-col gap-2 rounded-lg bg-muted/60 p-3">
      <p className="font-serif text-[0.95rem] leading-relaxed">
        {source.ru_summary}
      </p>
      {marks.length ? (
        <div className="flex flex-wrap gap-1.5">
          {marks.map((mark) => (
            <Badge key={mark} variant="outline">
              <SparklesIcon data-icon="inline-start" />
              {mark}
            </Badge>
          ))}
        </div>
      ) : null}
    </div>
  )
}

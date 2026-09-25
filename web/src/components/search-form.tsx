import { useState } from "react"
import { SearchIcon } from "lucide-react"
import { Link } from "react-router"

import { ApiError, activeRunId } from "@/api/client"
import { EXAMPLE_QUERIES, useStartSearch } from "@/api/queries"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group"
import { Spinner } from "@/components/ui/spinner"

const MIN_LEN = 3
const MAX_LEN = 300

export function SearchForm({
  initialQuery = "",
  autoFocus = false,
}: {
  initialQuery?: string
  autoFocus?: boolean
}) {
  const [query, setQuery] = useState(initialQuery)
  const search = useStartSearch()
  const trimmed = query.trim()
  const tooShort = trimmed.length > 0 && trimmed.length < MIN_LEN

  function submit(value: string) {
    const q = value.trim()
    if (q.length < MIN_LEN || search.isPending) return
    search.mutate(q)
  }

  return (
    <div className="flex flex-col gap-3">
      <form
        role="search"
        onSubmit={(event) => {
          event.preventDefault()
          submit(query)
        }}
      >
        <InputGroup className="h-14 rounded-xl bg-card shadow-xs">
          <InputGroupAddon>
            <SearchIcon />
          </InputGroupAddon>
          <InputGroupInput
            name="query"
            aria-label="Технологическое направление"
            placeholder="Например, перспективные решения в финтехе…"
            value={query}
            maxLength={MAX_LEN}
            autoFocus={autoFocus}
            autoComplete="off"
            aria-invalid={tooShort || undefined}
            onChange={(event) => setQuery(event.target.value)}
            className="text-base"
          />
          <InputGroupAddon align="inline-end">
            <InputGroupButton
              type="submit"
              variant="default"
              size="sm"
              className="h-10 px-4"
              disabled={trimmed.length < MIN_LEN || search.isPending}
            >
              {search.isPending ? <Spinner data-icon="inline-start" /> : null}
              Найти сигналы
            </InputGroupButton>
          </InputGroupAddon>
        </InputGroup>
      </form>
      {tooShort ? (
        <p className="text-sm text-muted-foreground">
          Запрос должен быть не короче трёх символов.
        </p>
      ) : null}
      {search.error ? <SearchError error={search.error} /> : null}
      <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        <span>Попробуйте:</span>
        {EXAMPLE_QUERIES.map((example) => (
          <button
            key={example}
            type="button"
            className="rounded-md border bg-card px-2.5 py-1 text-foreground transition-colors hover:border-primary/40 hover:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
            onClick={() => {
              setQuery(example)
              submit(example)
            }}
            disabled={search.isPending}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  )
}

function SearchError({ error }: { error: Error }) {
  const running = error instanceof ApiError ? activeRunId(error) : null
  if (running) {
    return (
      <Alert>
        <AlertTitle>Сейчас идёт другой поиск</AlertTitle>
        <AlertDescription>
          На ноутбуке поиск выполняется по одному: языковая модель одна.
          Дождитесь окончания или{" "}
          <Link
            to={`/run/${running}`}
            className="font-medium text-primary underline underline-offset-4"
          >
            откройте идущий поиск
          </Link>
          .
        </AlertDescription>
      </Alert>
    )
  }
  return (
    <Alert variant="destructive">
      <AlertTitle>Поиск не запустился</AlertTitle>
      <AlertDescription>{error.message}</AlertDescription>
    </Alert>
  )
}

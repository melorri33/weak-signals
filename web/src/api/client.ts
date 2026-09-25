/**
 * Клиент к FastAPI. Типы берутся из schema.d.ts — он сгенерирован из OpenAPI бэкенда
 * (`npm run gen:api`), поэтому контракт из src/common/schemas.py не переписывается руками.
 *
 * Запросы идут на /api: в разработке их проксирует Vite, на стенде — nginx.
 */
import type { components } from "./schema"

type Schemas = components["schemas"]

export type SearchResult = Schemas["SearchResult"]
export type SignalCard = Schemas["SignalCard"]
export type SourceRef = Schemas["SourceRef"]
export type Explanation = Schemas["Explanation"]
export type FilterDecision = Schemas["FilterDecision"]
export type ScoredCandidate = Schemas["ScoredCandidate"]
export type ModelCall = Schemas["ModelCall"]
export type SourceFailure = Schemas["SourceFailure"]
export type RunSummary = Schemas["RunSummary"]
export type Health = Schemas["Health"]
export type TrustLevel = Schemas["TrustLevel"]
export type SourceType = Schemas["SourceType"]
export type ReasonCode = FilterDecision["reason_code"]

export const API_BASE = "/api"

/** Ошибка API с текстом от сервера: сервер отвечает по-русски, показываем как есть. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    })
  } catch {
    throw new ApiError(
      0,
      "Сервер не отвечает. Проверьте, что запущен api: docker compose up -d api"
    )
  }
  if (!response.ok) {
    throw new ApiError(response.status, await errorText(response))
  }
  return (await response.json()) as T
}

async function errorText(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === "string") return body.detail
    if (Array.isArray(body.detail))
      return "Запрос должен быть от 3 до 300 символов"
  } catch {
    // тело не json — ниже общий текст
  }
  return `Сервер ответил ошибкой ${response.status}`
}

export const api = {
  startSearch: (query: string) =>
    request<SearchResult>("/search", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),
  getRun: (runId: string) =>
    request<SearchResult>(`/search/${encodeURIComponent(runId)}`),
  listRuns: () => request<RunSummary[]>("/runs"),
  health: () => request<Health>("/health"),
}

/** Из текста 429 «Уже идёт прогон abc — …» достаём id, чтобы дать ссылку на идущий прогон. */
export function activeRunId(error: ApiError): string | null {
  if (error.status !== 429) return null
  return /прогон\s+([0-9a-f]{6,})/i.exec(error.message)?.[1] ?? null
}

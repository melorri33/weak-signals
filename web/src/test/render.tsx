import type { ReactElement } from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render } from "@testing-library/react"
import { RouterProvider, createMemoryRouter } from "react-router"

import type { SearchResult } from "@/api/client"
import { TooltipProvider } from "@/components/ui/tooltip"
import fixture from "./fixtures/run.json"

/** В ответе API списки со значением по умолчанию необязательны; в фикстуре они есть всегда. */
export type FullRun = SearchResult &
  Required<
    Pick<SearchResult, "top" | "excluded" | "scored" | "source_failures">
  >

/** Прогон из фикстуры бэкенда (tests/fixtures/search_result_example.json) + scored и source_failures. */
export function exampleRun(): FullRun {
  return structuredClone(fixture) as FullRun
}

/** Рендер с теми же провайдерами, что в приложении; маршруты — в памяти. */
export function renderWithApp(
  ui: ReactElement,
  { path = "/", route = "/" }: { path?: string; route?: string } = {}
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const router = createMemoryRouter([{ path, element: ui }], {
    initialEntries: [route],
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <RouterProvider router={router} />
      </TooltipProvider>
    </QueryClientProvider>
  )
}

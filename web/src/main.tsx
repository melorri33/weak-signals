import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { RouterProvider, createBrowserRouter } from "react-router"

import "./index.css"
import { AppShell } from "@/components/app-shell"
import { ThemeProvider } from "@/components/theme-provider"
import { TooltipProvider } from "@/components/ui/tooltip"
import { HomePage } from "@/pages/home-page"
import { NotFoundPage } from "@/pages/not-found-page"

const queryClient = new QueryClient({
  defaultOptions: {
    // Готовый прогон не меняется: повторно не запрашиваем, при ошибке сети — одна повторная попытка.
    queries: { staleTime: 30_000, retry: 1, refetchOnWindowFocus: false },
  },
})

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: "/", element: <HomePage /> },
      // Прогон и инсайт грузятся отдельно: главная открывается быстрее, markdown нужен только в инсайте.
      {
        path: "/run/:runId",
        lazy: async () => ({
          Component: (await import("@/pages/run-page")).RunPage,
        }),
      },
      {
        path: "/run/:runId/signal/:candidateId",
        lazy: async () => ({
          Component: (await import("@/pages/insight-page")).InsightPage,
        }),
      },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
])

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider delay={200}>
          <RouterProvider router={router} />
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>
)

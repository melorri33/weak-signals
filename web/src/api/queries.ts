/** Запросы к API через TanStack Query: опрос идущего прогона и запуск поиска. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router"

import { api } from "@/api/client"

const POLL_MS = 2000

/** Примеры — формулировки из ТЗ, а не темы датасета: поиск не должен подсказывать ответ. */
export const EXAMPLE_QUERIES = [
  "перспективные решения в финтехе",
  "технологии в ИИ",
  "слабые сигналы в области кибербезопасности",
]

export function useStartSearch() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.startSearch,
    onSuccess: (result) => {
      queryClient.setQueryData(["run", result.run_id], result)
      void queryClient.invalidateQueries({ queryKey: ["runs"] })
      navigate(`/run/${result.run_id}`)
    },
  })
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    // Пока прогон идёт — опрашиваем: шаг и счётчики меняются на глазах. Готовый прогон не меняется.
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? POLL_MS : false,
    staleTime: (query) =>
      query.state.data?.status === "running" ? 0 : Infinity,
  })
}

/**
 * Признаки кандидатов для карты и графиков. Появляются на шаге «считаем признаки»,
 * поэтому у идущего прогона опрашиваем, пока не придут; у старых прогонов их нет (null).
 */
export function useEvidence(runId: string, running: boolean) {
  return useQuery({
    queryKey: ["evidence", runId],
    queryFn: () => api.getEvidence(runId),
    refetchInterval: (query) =>
      running && !query.state.data ? POLL_MS * 2 : false,
    staleTime: (query) => (query.state.data ? Infinity : 0),
    enabled: Boolean(runId),
  })
}

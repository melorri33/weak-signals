import { Badge } from "@/components/ui/badge"
import { Spinner } from "@/components/ui/spinner"

export function RunStatusBadge({
  status,
  stage,
}: {
  status: "running" | "done" | "error"
  stage: string
}) {
  if (status === "done") return <Badge variant="high">Готово</Badge>
  if (status === "error") return <Badge variant="low">Ошибка</Badge>
  return (
    <Badge variant="secondary">
      <Spinner data-icon="inline-start" />
      {stage || "идёт"}
    </Badge>
  )
}

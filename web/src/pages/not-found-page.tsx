import { Link } from "react-router"

import { Button } from "@/components/ui/button"
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty"

export function NotFoundPage({
  title = "Страница не найдена",
  description,
}: {
  title?: string
  description?: string
}) {
  return (
    <Empty className="border py-16">
      <EmptyHeader>
        <EmptyTitle>{title}</EmptyTitle>
        <EmptyDescription>
          {description ?? "Проверьте адрес или начните новый поиск."}
        </EmptyDescription>
      </EmptyHeader>
      <EmptyContent>
        <Button render={<Link to="/" />}>Новый поиск</Button>
      </EmptyContent>
    </Empty>
  )
}

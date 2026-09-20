"""Реестр прогонов: запуск конвейера в фоне и снимок его состояния по run_id.

Полный прогон на ноутбуке идёт минуты, поэтому POST /search не ждёт результата: он заводит прогон
и сразу отдаёт run_id со status='running'. Интерфейс опрашивает GET /search/{run_id} и показывает
шаг из stage. Снимок берётся из памяти, а если прогон делал другой процесс (или сервер перезапустили) —
из базы через storage.get_search_result.

Одновременных прогонов — один: локальная модель на ноутбуке одна, и два прогона просто уводят
машину в swap (проверено на 16 ГБ: две модели в памяти дали load average 32).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from uuid import uuid4

from src.common.logs import get_logger
from src.common.schemas import SearchResult
from src.pipeline import deps
from src.pipeline.run import run as run_pipeline

log = get_logger(__name__)

MAX_ACTIVE_RUNS = 1
# Сколько последних прогонов держим в памяти: без базы это единственное место, где они живут.
KEEP_RESULTS = 20

STAGE_QUEUED = "в очереди"


class TooManyRuns(RuntimeError):
    """Прогон уже идёт: место в очереди освободится, когда он закончится."""


class RunRegistry:
    """Прогоны этого процесса: активные задачи и снимки результатов."""

    def __init__(self, max_active: int = MAX_ACTIVE_RUNS, keep: int = KEEP_RESULTS) -> None:
        self._max_active = max_active
        self._keep = keep
        self._snapshots: OrderedDict[str, SearchResult] = OrderedDict()
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def active(self) -> list[str]:
        return [run_id for run_id, task in self._tasks.items() if not task.done()]

    def start(self, query: str) -> SearchResult:
        """Завести прогон и вернуть его начальное состояние. Уже идёт другой — TooManyRuns."""
        self._forget_finished()
        if len(self.active) >= self._max_active:
            raise TooManyRuns(f"Уже идёт прогон {self.active[0]} — дождитесь его окончания")
        run_id = uuid4().hex[:12]
        result = SearchResult(run_id=run_id, query=query, stage=STAGE_QUEUED)
        self._remember(result)
        self._tasks[run_id] = asyncio.create_task(self._execute(run_id, query))
        log.info("Прогон %s запущен: «%s»", run_id, query)
        return result

    def get(self, run_id: str) -> SearchResult | None:
        """Снимок прогона: из памяти, иначе из базы."""
        snapshot = self._snapshots.get(run_id)
        return snapshot if snapshot is not None else deps.get_search_result(run_id)

    async def cancel_all(self) -> None:
        """Остановить прогоны при выключении сервера, чтобы не оставлять висящих задач."""
        pending = [task for task in self._tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def _execute(self, run_id: str, query: str) -> None:
        try:
            self._remember(await run_pipeline(query, run_id=run_id, on_progress=self._remember))
        except asyncio.CancelledError:
            self._fail(run_id, "Прогон остановлен")
            raise
        except Exception as exc:  # конвейер не должен уронить сервер: ошибка видна в GET /search
            log.exception("Прогон %s упал", run_id)
            self._fail(run_id, f"{type(exc).__name__}: {exc}")

    def _remember(self, result: SearchResult) -> None:
        """Запомнить состояние прогона.

        Конвейер отдаёт один и тот же объект на каждом шаге и меняет его на месте — это и нужно:
        GET /search/{run_id} показывает текущий шаг, а не тот, что был при первом сохранении.
        """
        self._snapshots[result.run_id] = result
        self._snapshots.move_to_end(result.run_id)

    def _fail(self, run_id: str, error: str) -> None:
        result = self._snapshots.get(run_id)
        if result is None:
            return
        result.status = "error"
        result.error = error

    def _forget_finished(self) -> None:
        for run_id, task in list(self._tasks.items()):
            if task.done():
                self._tasks.pop(run_id)
        while len(self._snapshots) > self._keep:
            self._snapshots.popitem(last=False)

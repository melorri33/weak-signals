"""FastAPI поверх конвейера: запуск поиска, состояние прогона, карточка сигнала, проверка готовности.

Эндпоинты из контракта (CLAUDE.md):
    POST /search                              → запускает прогон, сразу отдаёт run_id
    GET  /search/{run_id}                     → SearchResult: пока идёт — со stage, потом с топ-15
    GET  /signal/{run_id}/{candidate_id}      → карточка одного сигнала
    GET  /runs                                → последние прогоны (память + data/search_runs)
    GET  /health                              → доступна ли модель и база

Запуск: uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.runs import RunRegistry, TooManyRuns
from src.api.saved_runs import RunSummary
from src.common.config import get_settings
from src.common.logs import get_logger
from src.common.schemas import SearchResult, SignalCard
from src.llm.client import LLMClient
from src.pipeline import deps

log = get_logger(__name__)

MIN_QUERY_LEN = 3
MAX_QUERY_LEN = 300

registry = RunRegistry()


class SearchRequest(BaseModel):
    """Свободный запрос пользователя — тот же, что в CLI."""

    query: str = Field(min_length=MIN_QUERY_LEN, max_length=MAX_QUERY_LEN)


class Health(BaseModel):
    """Что сейчас работает. Без модели поиск бессмысленен, без базы — просто не переживёт перезапуск."""

    status: Literal["ok", "degraded"]
    llm_model: str
    llm_available: bool
    database_available: bool
    active_runs: list[str]
    notes: list[str] = Field(default_factory=list, description="что не готово, по-русски")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await registry.cancel_all()


app = FastAPI(
    title="Слабые сигналы",
    description="Поиск зарождающихся технологий по открытым источникам: запрос → топ-15 сигналов.",
    lifespan=lifespan,
)


# async def, а не def: registry.start заводит фоновую задачу, а для этого нужен рабочий цикл событий —
# синхронный обработчик FastAPI выполняет в отдельном потоке, где цикла нет.
@app.post("/search", response_model=SearchResult, status_code=202)
async def start_search(request: SearchRequest) -> SearchResult:
    """Запустить прогон. Ответ приходит сразу: результат забирается по GET /search/{run_id}."""
    try:
        return registry.start(request.query.strip())
    except TooManyRuns as exc:
        # 429, а не 503: запрос корректный, просто ноутбук делает один прогон за раз.
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.get("/search/{run_id}", response_model=SearchResult)
def get_search(run_id: str) -> SearchResult:
    """Состояние прогона: stage — текущий шаг, status='done' — выдача готова."""
    result = registry.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Прогон {run_id} не найден")
    return result


@app.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    """Последние прогоны для списка в интерфейсе: идущий и готовые, свежие первыми."""
    return registry.summaries()


@app.get("/signal/{run_id}/{candidate_id}", response_model=SignalCard)
def get_signal(run_id: str, candidate_id: str) -> SignalCard:
    """Карточка одного сигнала из выдачи прогона."""
    result = registry.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Прогон {run_id} не найден")
    card = next((c for c in result.top if c.candidate_id == candidate_id), None)
    if card is None:
        raise HTTPException(status_code=404, detail=f"В прогоне {run_id} нет сигнала {candidate_id}")
    return card


@app.get("/health", response_model=Health)
async def health() -> Health:
    """Готовность сервиса: модель и база. Полезно перед демонстрацией — сразу видно, что не поднято."""
    settings = get_settings()
    notes: list[str] = []
    try:
        llm_available = await LLMClient.from_settings().is_available()
    except Exception as exc:
        log.warning("проверка модели не удалась: %s", exc)
        llm_available = False
    if not llm_available:
        notes.append(
            f"Модель {settings.llm_model} недоступна: проверьте `ollama serve` и `ollama pull {settings.llm_model}`"
        )
    # to_thread: проверка базы блокирующая (до 2 с на таймаут подключения), а цикл событий занят прогоном.
    database_available = await asyncio.to_thread(deps.database_ok)
    if not database_available:
        notes.append("База недоступна: прогоны живут только в памяти и не переживут перезапуск сервера")
    return Health(
        status="ok" if llm_available and database_available else "degraded",
        llm_model=settings.llm_model,
        llm_available=llm_available,
        database_available=database_available,
        active_runs=registry.active,
        notes=notes,
    )

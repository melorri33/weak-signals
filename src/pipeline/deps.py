"""Подключение чужих модулей к конвейеру, пока они ещё не готовы.

Правило простое: если модуль владельца уже есть — работаем с ним, если нет — подставляем
безопасную заглушку и громко пишем об этом в лог. Так конвейер собирается целиком уже сегодня,
а замена заглушки на настоящий модуль не требует правок в run.py.

Что чем подменяется:
    collectors.collect     → документы из tests/fixtures/documents_example.json
                             (настоящий сбор — без самопубликаций, см. src/pipeline/self_published.py)
    collectors.term_stats  → None (features.compute умеет работать без статистики)
    storage.save_documents → ничего не делаем
    storage.get_search_result → None (API отдаёт прогон из памяти)
    trust.level_for        → TrustLevel.MEDIUM
    filters.apply          → «ok», никого не исключаем
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.common.logs import get_logger
from src.common.schemas import (
    Candidate,
    CandidateFeatures,
    Document,
    FilterDecision,
    SearchResult,
    SourceType,
    TermStats,
    TrustLevel,
)
from src.pipeline.self_published import without_self_published

log = get_logger(__name__)

FIXTURE_DOCUMENTS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "documents_example.json"

# Типы источников, которые по ТЗ сами по себе не могут быть основанием для сигнала.
WEAK_SOURCE_TYPES = {SourceType.SOCIAL, SourceType.BLOG, SourceType.PRESS_RELEASE}


def _load(path: str, name: str) -> Any | None:
    """Достать функцию name из модуля path, если владелец её уже написал."""
    try:
        module = __import__(path, fromlist=[name])
    except ImportError:
        return None
    return getattr(module, name, None)


@lru_cache
def _warn_once(what: str) -> None:
    log.warning("%s ещё не готов — работаю на заглушке", what)


async def collect(phrases: list[str], limit: int) -> list[Document]:
    """Собрать документы по поисковым фразам (collectors.collect), без самопубликаций."""
    real: Callable[..., Any] | None = _load("src.collectors", "collect")
    if real is None:
        _warn_once("collectors.collect")
        return fixture_documents()[:limit]
    return without_self_published(await real(phrases, limit=limit))


def new_run() -> None:
    """Начало прогона: сборщик забывает, какие ленты «исчерпались» в прошлых прогонах.

    Лента, ответившая 429, выключается до конца прогона (collectors.news), но сброса между прогонами
    никто не вызывал — в API и в прогоне по шести областям подряд лента оставалась выключенной
    до перезапуска процесса. 26.09: SiliconANGLE ответил 429 на десятой секунде первой области
    и дальше не дал ни одного документа — 15 ответов за все шесть областей против ~250 на область
    со сбросом. VentureBeat — ноль.
    """
    forget: Callable[..., Any] | None = _load("src.collectors.news", "forget_exhausted_feeds")
    if forget is not None:
        forget()


def fixture_documents() -> list[Document]:
    """Примеры документов для офлайн-прогонов и тестов."""
    raw = json.loads(FIXTURE_DOCUMENTS.read_text(encoding="utf-8"))
    return [Document.model_validate(item) for item in raw]


async def term_stats(term: str) -> TermStats | None:
    """Статистика по термину (collectors.term_stats). None — статистики нет, признаки считаются без неё."""
    real: Callable[..., Any] | None = _load("src.collectors", "term_stats")
    if real is None:
        _warn_once("collectors.term_stats")
        return None
    return await real(term)


def save_documents(docs: list[Document]) -> None:
    """Сохранить документы (storage.save_documents)."""
    real: Callable[..., Any] | None = _load("src.storage", "save_documents")
    if real is None:
        _warn_once("storage.save_documents")
        return
    real(docs)


def save_search_result(result: SearchResult) -> None:
    """Сохранить промежуточный или итоговый результат прогона (storage.save_search_result)."""
    real: Callable[..., Any] | None = _load("src.storage", "save_search_result")
    if real is None:
        _warn_once("storage.save_search_result")
        return
    real(result)


def trust_level(doc: Document) -> TrustLevel:
    """Уровень доверия к источнику (trust.level_for)."""
    real: Callable[..., Any] | None = _load("src.trust", "level_for")
    if real is None:
        _warn_once("trust.level_for")
        return TrustLevel.LOW if doc.source_type in WEAK_SOURCE_TYPES else TrustLevel.MEDIUM
    return real(doc.url, doc.source_type)


def filter_decision(candidate: Candidate, features: CandidateFeatures) -> FilterDecision:
    """Решение правил отсева (filters.apply)."""
    real: Callable[..., Any] | None = _load("src.filters", "apply")
    if real is None:
        _warn_once("filters.apply")
        return FilterDecision(
            candidate_id=candidate.id,
            name=candidate.name,
            excluded=False,
            reason_code="ok",
            reason_text="Правила отсева ещё не подключены",
        )
    return real(candidate, features)


def get_search_result(run_id: str) -> SearchResult | None:
    """Достать сохранённый прогон (storage.get_search_result). None — такого прогона в базе нет."""
    real: Callable[..., Any] | None = _load("src.storage", "get_search_result")
    if real is None:
        _warn_once("storage.get_search_result")
        return None
    return real(run_id)


def database_ok() -> bool:
    """Отвечает ли база: нужно для GET /health, чтобы сразу видеть, переживёт ли выдача перезапуск."""
    session_factory: Callable[..., Any] | None = _load("src.storage.db", "get_session")
    if session_factory is None:
        return False
    try:
        from sqlalchemy import text

        with session_factory() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        log.warning("база не ответила: %s", exc)
        return False
    return True

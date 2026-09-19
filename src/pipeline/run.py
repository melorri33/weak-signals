"""Конвейер: `run(query) -> SearchResult`.

Шесть шагов из ТЗ, у каждого свой бюджет времени. Правила:
  - шаг упал или не успел — пишем в лог и идём дальше с тем, что есть;
  - без документов сигналов быть не может, поэтому пустой сбор источников — это status='error';
  - после каждого шага обновляем stage и сохраняем результат: по нему UI показывает прогресс;
  - все вызовы моделей попадают в SearchResult.model_calls (требование ТЗ).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar
from uuid import uuid4

from src.common.config import get_settings
from src.common.logs import collected_model_calls, get_logger, start_run_log
from src.common.schemas import (
    CONFIDENT_THRESHOLD,
    TOP_N,
    Candidate,
    CandidateFeatures,
    Document,
    ScoredCandidate,
    SearchResult,
    SignalCard,
)
from src.features import compute
from src.llm.cards import NoSourcesError, make_card
from src.llm.expand_query import expand_query
from src.model import score
from src.pipeline import deps
from src.pipeline.candidates import extract_candidates

log = get_logger(__name__)

T = TypeVar("T")

# Бюджеты шагов в секундах. Сумма — примерно 4 минуты на прогон.
EXPAND_BUDGET_S = 15.0
CANDIDATES_BUDGET_S = 40.0
STATS_BUDGET_S = 60.0
SCORE_BUDGET_S = 5.0
CARDS_BUDGET_S = 60.0

# Сколько запросов статистики и карточек делаем одновременно.
STATS_CONCURRENCY = 8
CARDS_CONCURRENCY = 3

STAGE_START = "начинаем"
STAGE_DONE = "готово"


async def run(
    query: str,
    *,
    run_id: str | None = None,
    on_progress: Callable[[SearchResult], None] | None = None,
) -> SearchResult:
    """Прогнать запрос через весь конвейер и вернуть готовую выдачу."""
    start_run_log()
    started = time.perf_counter()
    settings = get_settings()
    result = SearchResult(run_id=run_id or uuid4().hex[:12], query=query, stage=STAGE_START)

    def progress(stage: str) -> None:
        result.stage = stage
        result.model_calls = collected_model_calls()
        deps.save_search_result(result)
        if on_progress is not None:
            on_progress(result)

    progress("расширяем запрос")
    phrases = await _with_budget("expand_query", EXPAND_BUDGET_S, expand_query(query), default=[])
    result.expanded_phrases = phrases or [query]

    progress("собираем источники")
    docs = await _with_budget(
        "collect",
        settings.collect_budget_s,
        deps.collect(result.expanded_phrases, limit=settings.max_documents),
        default=[],
    )
    if not docs:
        return _failed(result, started, "Источники ничего не вернули — сигналы строить не на чем")
    for doc in docs:
        doc.trust = deps.trust_level(doc)
    deps.save_documents(docs)
    result.documents_processed = len(docs)

    progress("выделяем кандидатов")
    candidates = await _with_budget("extract_candidates", CANDIDATES_BUDGET_S, extract_candidates(docs), default=[])
    if not candidates:
        return _failed(result, started, "Не удалось выделить ни одной технологии-кандидата")
    result.candidates_found = len(candidates)

    progress("считаем признаки")
    features = await _features(candidates, docs)

    progress("отсев и скоринг")
    kept, features_kept = _apply_filters(candidates, features, result)
    scored = _score(kept, features_kept)

    progress("собираем карточки")
    result.top = await _cards(scored[:TOP_N], candidates, docs)
    result.confident_signals = sum(card.score > CONFIDENT_THRESHOLD for card in result.top)

    result.status = "done"
    result.duration_s = round(time.perf_counter() - started, 1)
    progress(STAGE_DONE)
    log.info(
        "Прогон %s: документов %d, кандидатов %d, в выдаче %d, %.1f с",
        result.run_id,
        result.documents_processed,
        result.candidates_found,
        len(result.top),
        result.duration_s,
    )
    return result


async def _features(candidates: list[Candidate], docs: list[Document]) -> list[CandidateFeatures]:
    """Статистика по каждому кандидату (параллельно) и признаки по ней."""
    semaphore = asyncio.Semaphore(STATS_CONCURRENCY)

    async def stats_for(candidate: Candidate) -> CandidateFeatures:
        async with semaphore:
            try:
                stats = await deps.term_stats(candidate.name)
            except Exception as exc:
                log.warning("term_stats(%s) не отработал: %s", candidate.name, exc)
                stats = None
        return compute(candidate, docs, stats)

    gathered = await _with_budget(
        "term_stats",
        STATS_BUDGET_S,
        asyncio.gather(*(stats_for(c) for c in candidates)),
        default=[],
    )
    if gathered:
        return list(gathered)
    log.warning("Статистика не успела в бюджет — считаю признаки только по документам")
    return [compute(c, docs, None) for c in candidates]


def _apply_filters(
    candidates: list[Candidate],
    features: list[CandidateFeatures],
    result: SearchResult,
) -> tuple[list[Candidate], list[CandidateFeatures]]:
    """Отсев зрелого, хайпа и шума. Исключённые с причиной уходят в SearchResult.excluded."""
    by_id = {f.candidate_id: f for f in features}
    kept: list[Candidate] = []
    kept_features: list[CandidateFeatures] = []
    for candidate in candidates:
        feature = by_id.get(candidate.id) or compute(candidate, [], None)
        try:
            decision = deps.filter_decision(candidate, feature)
        except Exception as exc:
            log.warning("filters.apply(%s) упал: %s — кандидата оставляю", candidate.id, exc)
            kept.append(candidate)
            kept_features.append(feature)
            continue
        if decision.excluded:
            result.excluded.append(decision)
            continue
        kept.append(candidate)
        kept_features.append(feature)
    log.info("После отсева осталось %d кандидатов, исключено %d", len(kept), len(result.excluded))
    return kept, kept_features


def _score(candidates: list[Candidate], features: list[CandidateFeatures]) -> list[ScoredCandidate]:
    """Скоринг модели. Упал — прогон продолжаем без ранжирования."""
    if not candidates:
        return []
    try:
        return score(candidates, features)
    except Exception:
        log.exception("model.score упал — выдача будет без уверенности модели")
        return []


async def _cards(
    scored: list[ScoredCandidate],
    candidates: list[Candidate],
    docs: list[Document],
) -> list[SignalCard]:
    """Карточки топа. Кандидат без проверенных источников в выдачу не идёт (требование ТЗ)."""
    by_id = {c.id: c for c in candidates}
    docs_by_id = {d.id: d for d in docs}
    semaphore = asyncio.Semaphore(CARDS_CONCURRENCY)

    async def card_for(item: ScoredCandidate) -> SignalCard | None:
        candidate = by_id.get(item.candidate_id)
        own_docs = [docs_by_id[i] for i in (candidate.document_ids if candidate else []) if i in docs_by_id]
        async with semaphore:
            try:
                return await make_card(item, own_docs, name_ru=candidate.name_ru if candidate else None)
            except NoSourcesError as exc:
                log.warning("Карточка не собрана: %s", exc)
            except Exception:
                log.exception("Карточка для %s не собралась", item.candidate_id)
        return None

    cards = await _with_budget(
        "make_card",
        CARDS_BUDGET_S,
        asyncio.gather(*(card_for(s) for s in scored)),
        default=[],
    )
    return [c for c in cards if c is not None]


async def _with_budget(step: str, budget_s: float, work: Awaitable[T], default: T) -> T:
    """Выполнить шаг в рамках бюджета. Не успел или упал — вернуть default и жить дальше."""
    try:
        async with asyncio.timeout(budget_s):
            return await work
    except TimeoutError:
        log.error("Шаг %s не успел за %.0f с — идём дальше", step, budget_s)
    except Exception:
        log.exception("Шаг %s упал — идём дальше", step)
    return default


def _failed(result: SearchResult, started: float, message: str) -> SearchResult:
    """Прогон дальше не имеет смысла: фиксируем ошибку понятным текстом."""
    result.status = "error"
    result.stage = "ошибка"
    result.error = message
    result.duration_s = round(time.perf_counter() - started, 1)
    result.model_calls = collected_model_calls()
    deps.save_search_result(result)
    log.error("Прогон %s остановлен: %s", result.run_id, message)
    return result

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
from collections import deque
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
    TermStats,
)
from src.features import compute
from src.llm.cards import NoSourcesError, make_card
from src.llm.expand_query import expand_query
from src.model import score
from src.pipeline import deps
from src.pipeline.candidates import extract_candidates

log = get_logger(__name__)

T = TypeVar("T")

# Бюджеты шагов в секундах. Замеры на MacBook Air M1 16 ГБ: расширение запроса на qwen3:8b занимает
# 60–92 с (таблица в README), поэтому изначальные 15 с оказались нереальными. Это один вызов на прогон,
# и от его качества зависит, найдём ли мы технологию вообще, — поэтому бюджет щедрый.
# Скоринг синхронный и занимает миллисекунды, отдельный бюджет ему не нужен.
EXPAND_BUDGET_S = 120.0
# Кандидатов выделяет LLM пачками документов. На видеокарте (RTX 3060, qwen3:14b, 32 токена/с)
# пачка идёт секунды, а не минуты. Не успели — берём то, что уже выписано, и идём дальше.
CANDIDATES_BUDGET_S = 180.0
STATS_BUDGET_S = 60.0
# Пятнадцать карточек, каждая — отдельный вызов модели. Замер на RTX 3060 с qwen3:14b:
# 15 вызовов за 150 с, то есть около 10 с на карточку. Пятнадцать карточек — это пять пачек
# по CARDS_CONCURRENCY, примерно 190 с; остальное — запас на кандидатов, у которых карточка
# не собралась и нужен добор следующего. На 150 с шаг упирался в потолок и отдавал 12 из 15.
CARDS_BUDGET_S = 300.0

# Уверенность запасного ранжирования, когда модель не отработала: ни за, ни против.
NEUTRAL_SCORE = 0.5

# Сколько кандидатов сразу за топ-15 показываем в логе прогона.
NEAR_MISSES_IN_LOG = 15

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

    result.scored = scored

    progress("собираем карточки")
    _log_near_misses(scored)
    result.top = await _cards(scored, candidates, docs)
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
    """Статистика по каждому кандидату и признаки по ней.

    Бюджет общий на все кандидаты, но результат частичный: кто успел — тот со статистикой,
    остальным считаем признаки по найденным документам. Раньше на таймауте терялась статистика
    всех кандидатов сразу, и модель ранжировала вслепую.

    Про лимиты источников (выяснили при обучении модели): техмедиа отвечают 403 на параллельные
    запросы и их приходится звать очередью, поэтому 50 кандидатов могут не влезть в бюджет.
    Это внутри term_stats (модуль Данных) — здесь мы лишь ограничиваем число одновременных вызовов
    и переживаем недобор.
    """
    semaphore = asyncio.Semaphore(STATS_CONCURRENCY)

    async def stats_for(candidate: Candidate) -> tuple[str, TermStats | None]:
        async with semaphore:
            try:
                return candidate.id, await deps.term_stats(candidate.name)
            except Exception as exc:
                log.warning("term_stats(%s) не отработал: %s", candidate.name, exc)
                return candidate.id, None

    started = time.perf_counter()
    tasks = [asyncio.create_task(stats_for(c)) for c in candidates]
    done, pending = await asyncio.wait(tasks, timeout=STATS_BUDGET_S)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    log.info("Шаг term_stats: %.1f с из %.0f с бюджета", time.perf_counter() - started, STATS_BUDGET_S)

    stats_by_id: dict[str, TermStats | None] = {}
    for task in done:
        try:
            candidate_id, stats = task.result()
        except Exception:
            log.exception("Статистика по кандидату не получена")
            continue
        stats_by_id[candidate_id] = stats
    if pending:
        log.warning(
            "Статистика не успела для %d кандидатов из %d — их признаки считаю только по документам",
            len(pending),
            len(candidates),
        )
    return [compute(c, docs, stats_by_id.get(c.id)) for c in candidates]


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
    """Скоринг модели. Упал — ранжируем запасным способом, но выдачу не теряем."""
    if not candidates:
        return []
    try:
        return score(candidates, features)
    except Exception:
        log.exception("model.score упал — ранжирую по числу документов, уверенность нейтральная")
        return _ranking_by_documents(candidates)


def _ranking_by_documents(candidates: list[Candidate]) -> list[ScoredCandidate]:
    """Запасное ранжирование: чем больше документов у кандидата, тем выше. Уверенность нейтральная."""
    ordered = sorted(candidates, key=lambda c: len(c.document_ids), reverse=True)
    return [ScoredCandidate(candidate_id=c.id, name=c.name, score=NEUTRAL_SCORE) for c in ordered]


def _log_near_misses(scored: list[ScoredCandidate]) -> None:
    """Кандидаты, которые нашлись, но не попали в топ-15.

    Нужно, чтобы отличать «технологию не нашли» от «нашли, но ранжировали низко»: первое лечится
    источниками и выделением кандидатов, второе — скорингом и правилами отсева.
    """
    missed = scored[TOP_N : TOP_N + NEAR_MISSES_IN_LOG]
    if not missed:
        return
    log.info(
        "Не попали в топ-15 (ближайшие %d): %s",
        len(missed),
        "; ".join(f"{s.name} {s.score:.2f}" for s in missed),
    )


async def _cards(
    scored: list[ScoredCandidate],
    candidates: list[Candidate],
    docs: list[Document],
) -> list[SignalCard]:
    """Карточки топа. В выдаче должно быть ровно TOP_N: если карточка не собралась
    (нет проверенных источников или ошибка), добираем следующего кандидата по списку.

    Кандидат без источников в выдачу не идёт — это требование ТЗ, поэтому добор, а не заполнение
    пустышкой. Если кандидаты кончились раньше, честно возвращаем меньше и пишем в лог.

    Бюджет держим сами, пачка за пачкой, а не общим asyncio.timeout вокруг цикла: отмена приходила
    посреди пачки и уносила уже готовые карточки вместе с недоделанными — на прогоне 20.09 модель
    отработала 12 раз, а в выдачу попало 9. Теперь готовые карточки пачки забираем всегда.
    """
    by_id = {c.id: c for c in candidates}
    docs_by_id = {d.id: d for d in docs}
    queue = deque(scored)
    cards: list[SignalCard] = []

    async def card_for(item: ScoredCandidate) -> SignalCard | None:
        candidate = by_id.get(item.candidate_id)
        own_docs = [docs_by_id[i] for i in (candidate.document_ids if candidate else []) if i in docs_by_id]
        try:
            return await make_card(item, own_docs, name_ru=candidate.name_ru if candidate else None)
        except NoSourcesError as exc:
            log.warning("Карточка не собрана: %s — беру следующего кандидата", exc)
        except Exception:
            log.exception("Карточка для %s не собралась — беру следующего кандидата", item.candidate_id)
        return None

    started = time.perf_counter()
    deadline = started + CARDS_BUDGET_S
    while queue and len(cards) < TOP_N:
        batch_size = min(CARDS_CONCURRENCY, TOP_N - len(cards), len(queue))
        batch = [queue.popleft() for _ in range(batch_size)]
        tasks = [asyncio.create_task(card_for(item)) for item in batch]
        _, pending = await asyncio.wait(tasks, timeout=max(0.0, deadline - time.perf_counter()))
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # Идём по batch, а не по множеству done: порядок карточек — это порядок уверенности модели.
        cards.extend(t.result() for t in tasks if t not in pending and t.result() is not None)
        if pending:
            log.error("Шаг make_card не успел за %.0f с — отдаю %d карточек", CARDS_BUDGET_S, len(cards))
            break
    log.info("Шаг make_card: %.1f с из %.0f с бюджета", time.perf_counter() - started, CARDS_BUDGET_S)
    if len(cards) < TOP_N:
        log.warning(
            "В выдаче %d карточек вместо %d: кандидаты кончились или не собрались (осталось в очереди: %d)",
            len(cards),
            TOP_N,
            len(queue),
        )
    return cards


async def _with_budget(step: str, budget_s: float, work: Awaitable[T], default: T) -> T:
    """Выполнить шаг в рамках бюджета. Не успел или упал — вернуть default и жить дальше.

    Время каждого шага пишем в лог: по нему видно, где конвейер упирается в бюджет.
    """
    started = time.perf_counter()
    try:
        async with asyncio.timeout(budget_s):
            return await work
    except TimeoutError:
        log.error("Шаг %s не успел за %.0f с — идём дальше", step, budget_s)
    except Exception:
        log.exception("Шаг %s упал — идём дальше", step)
    finally:
        log.info("Шаг %s: %.1f с из %.0f с бюджета", step, time.perf_counter() - started, budget_s)
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

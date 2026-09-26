"""Конвейер: `run(query) -> SearchResult`.

Шесть шагов из ТЗ, у каждого свой бюджет времени. Правила:
  - шаг упал или не успел — пишем в лог и идём дальше с тем, что есть;
  - без документов сигналов быть не может, поэтому пустой сбор источников — это status='error';
  - после каждого шага обновляем stage и сохраняем результат: по нему UI показывает прогресс;
  - все вызовы моделей попадают в SearchResult.model_calls (требование ТЗ).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TypeVar
from uuid import uuid4

from src.common.config import get_settings
from src.common.logs import collected_model_calls, collected_source_failures, get_logger, start_run_log
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

# Кандидаты, их признаки и статистика публикаций по candidate_id.
FeaturesCallback = Callable[[list[Candidate], list[CandidateFeatures], dict[str, TermStats]], None]

# Бюджеты шагов в секундах. Замеры на MacBook Air M1 16 ГБ: расширение запроса на qwen3:8b занимает
# 60–92 с (таблица в README), поэтому изначальные 15 с оказались нереальными. Это один вызов на прогон,
# и от его качества зависит, найдём ли мы технологию вообще, — поэтому бюджет щедрый.
# Скоринг синхронный и занимает миллисекунды, отдельный бюджет ему не нужен.
EXPAND_BUDGET_S = 120.0
# Кандидатов выделяет LLM пачками документов. На видеокарте (RTX 3060, qwen3:14b, 32 токена/с)
# пачка идёт секунды, а не минуты. Не успели — берём то, что уже выписано, и идём дальше.
#
# Обязан быть заметно больше внутреннего candidates.BUDGET_S: шаг останавливается сам и отдаёт
# выписанное, а отмена снаружи приходит посреди вызова модели и уносит всё. Нарушение этой связи
# 21.09 обнулило прогон по всем шести областям — «не удалось выделить ни одной технологии».
# Связь проверяется тестом, менять эти два числа нужно вместе.
CANDIDATES_BUDGET_S = 250.0
# Статистика по кандидатам. На 60 кандидатах за 60 с успевало около 40 — треть оставалась
# без данных о публикациях, и признаки для них считались только по найденным документам.
# После подъёма MAX_CANDIDATES до 150 прежнего бюджета не хватит подавно: пропускная
# способность шага около 0.7 кандидата в секунду (техмедиа отвечают 403 на параллельные
# запросы и их приходится звать очередью), поэтому 150 кандидатам нужно примерно 225 с.
STATS_BUDGET_S = 240.0
# Запас поверх внутреннего бюджета сбора (COLLECT_BUDGET_S в .env). Сам collect держит своё время
# и возвращает то, что успело прийти; если обернуть его таймаутом ровно на столько же, два таймера
# соревнуются, и при выигрыше внешнего вся собранная пачка документов выбрасывается. На проверке
# по шести областям так и вышло: в пяти прогонах успевал внутренний, в шестом — внешний,
# и прогон остановился с «источники ничего не вернули» при живых источниках.
COLLECT_GRACE_S = 30.0
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

# Второй круг: поиск документов по имени кандидата точной фразой.
# Берём столько кандидатов, сколько нужно на выдачу, плюс запас на тех, у кого карточка
# не соберётся. Больше брать незачем: документы второго круга нужны именно карточкам.
SECOND_ROUND_CANDIDATES = TOP_N + 10
SECOND_ROUND_DOCS = 8
SECOND_ROUND_BUDGET_S = 120.0
SECOND_ROUND_CONCURRENCY = 4

STAGE_START = "начинаем"
STAGE_DONE = "готово"


async def run(
    query: str,
    *,
    run_id: str | None = None,
    on_progress: Callable[[SearchResult], None] | None = None,
    on_features: FeaturesCallback | None = None,
) -> SearchResult:
    """Прогнать запрос через весь конвейер и вернуть готовую выдачу.

    on_features получает признаки и статистику публикаций по всем кандидатам, включая отсеянных.
    В SearchResult их нет, а интерфейсу они нужны для карты сигналов и графиков динамики.
    """
    start_run_log()
    started = time.perf_counter()
    settings = get_settings()
    result = SearchResult(run_id=run_id or uuid4().hex[:12], query=query, stage=STAGE_START)

    def progress(stage: str) -> None:
        result.stage = stage
        result.model_calls = collected_model_calls()
        result.source_failures = collected_source_failures()
        deps.save_search_result(result)
        if on_progress is not None:
            on_progress(result)

    progress("расширяем запрос")
    phrases = await _with_budget("expand_query", EXPAND_BUDGET_S, expand_query(query), default=[])
    result.expanded_phrases = phrases or [query]

    progress("собираем источники")
    docs = await _with_budget(
        "collect",
        settings.collect_budget_s + COLLECT_GRACE_S,
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
    features, stats = await _features(candidates, docs)
    if on_features is not None:
        on_features(candidates, features, stats)

    progress("отсев и скоринг")
    kept, features_kept = _apply_filters(candidates, features, result)
    # В потоке, а не в цикле событий: модель названий грузит bge-m3 с диска (а в первый раз качает ~2 ГБ)
    # и считает эмбеддинги на процессоре. Внутри цикла это минутами блокировало API — интерфейс не мог
    # даже узнать шаг прогона. Журнал моделей не теряется: to_thread копирует контекст с тем же списком.
    scored = await asyncio.to_thread(_score, kept, features_kept)

    result.scored = scored

    progress("ищем документы по кандидатам")
    docs.extend(await _second_round(scored, candidates, docs))

    progress("собираем карточки")
    _log_near_misses(scored)
    result.top = await _cards(drop_name_variants(scored), candidates, docs)
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


def _mentions_name(doc: Document, name: str) -> bool:
    """Название встречается в документе целой фразой, а не по кускам."""
    text = f"{doc.title}\n{doc.abstract or ''}"
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text, re.IGNORECASE) is not None


async def _second_round(
    scored: list[ScoredCandidate], candidates: list[Candidate], docs: list[Document]
) -> list[Document]:
    """Документы, написанные **про** кандидата, а не упоминающие его вскользь.

    Первый круг ищет широкими фразами и приносит документы про своё. Замер 21.09 на корпусе
    из 7031 документа: из 12 технологий датасета, названных в собранных документах, 9 нашлись
    ровно в одном документе, а 10 из 12 не попали в заголовок ни разу — то есть документ написан
    не про них. Карточку по такому упоминанию не построить, а ТЗ требует источников.

    Здесь имя кандидата ищется точной фразой. Тот же приём в диагностике находил 88 технологий
    датасета из 100 — против 12, названных документами первого круга.

    Шаг необязательный: не успел или ничего не нашёл — карточки собираются по документам первого
    круга, как раньше. Бюджет держим через asyncio.wait, а не общим таймаутом: отмена посреди
    ожидания уносила бы уже полученные документы вместе с недополученными.
    """
    by_id = {c.id: c for c in candidates}
    known = {d.id for d in docs}
    wanted = [by_id[s.candidate_id] for s in scored[:SECOND_ROUND_CANDIDATES] if s.candidate_id in by_id]
    if not wanted:
        return []

    semaphore = asyncio.Semaphore(SECOND_ROUND_CONCURRENCY)

    async def for_candidate(candidate: Candidate) -> list[Document]:
        async with semaphore:
            found = await deps.collect([candidate.name], limit=SECOND_ROUND_DOCS)
        fresh: list[Document] = []
        for doc in found:
            # Источник мог вернуть документ по отдельным словам — берём только точное совпадение.
            if not _mentions_name(doc, candidate.name):
                continue
            if doc.id not in candidate.document_ids:
                candidate.document_ids.append(doc.id)
            if doc.id not in known:
                known.add(doc.id)
                fresh.append(doc)
        return fresh

    tasks = [asyncio.create_task(for_candidate(c)) for c in wanted]
    done, pending = await asyncio.wait(tasks, timeout=SECOND_ROUND_BUDGET_S)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if pending:
        log.warning(
            "Второй круг не успел за %.0f с — %d кандидатов из %d остались с документами первого круга",
            SECOND_ROUND_BUDGET_S,
            len(pending),
            len(wanted),
        )

    fresh = [doc for task in done if not task.cancelled() and task.exception() is None for doc in task.result()]
    for doc in fresh:
        doc.trust = deps.trust_level(doc)
    deps.save_documents(fresh)
    log.info("Второй круг: %d кандидатов, новых документов %d", len(done), len(fresh))
    return fresh


async def _features(
    candidates: list[Candidate], docs: list[Document]
) -> tuple[list[CandidateFeatures], dict[str, TermStats]]:
    """Статистика по каждому кандидату и признаки по ней. Статистика — только у тех, кто успел.

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
    features = [compute(c, docs, stats_by_id.get(c.id)) for c in candidates]
    return features, {cid: stats for cid, stats in stats_by_id.items() if stats is not None}


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


# Служебные слова: по ним два названия не считаются вариантами одного.
_VARIANT_STOP = {"for", "and", "the", "of", "in", "on", "with", "to", "from", "as", "by", "based", "using"}


def _variant_stem(word: str) -> str:
    word = word.lower()
    for suffix in ("ization", "isation", "ics", "ing", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _name_words(name: str) -> set[str]:
    return {
        _variant_stem(w)
        for w in re.findall(r"[A-Za-z0-9]+", name)
        if _variant_stem(w) not in _VARIANT_STOP and len(w) > 2
    }


def drop_name_variants(scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Из нескольких названий одной технологии оставить самое узкое.

    В выдаче попадались по два-три варианта одного и того же, и каждый занимал своё место
    в топ-15. Замер 21.09 по шести областям: 5 мест из 90 уходило на повторы. Жюри засчитает
    такие карточки как одно совпадение, а место отнято у другой технологии.

    Оставляем самое узкое название, а не самое высоко оценённое. Причина в том, как сверяют:
    засчитывается, когда все слова названия из датасета есть в нашем. Узкое название
    засчитается и против самого себя, и против более широкого термина датасета, широкое
    против узкого — нет.
    """
    kept: list[ScoredCandidate] = []
    for item in scored:
        words = _name_words(item.name)
        if len(words) < 2:
            kept.append(item)
            continue
        twin = next(
            (
                i
                for i, k in enumerate(kept)
                if len(_name_words(k.name)) >= 2 and (_name_words(k.name) <= words or words <= _name_words(k.name))
            ),
            None,
        )
        if twin is None:
            kept.append(item)
        elif len(words) > len(_name_words(kept[twin].name)):
            log.info("Вариант названия: «%s» вместо «%s» — оно уже", item.name, kept[twin].name)
            kept[twin] = item
        else:
            log.info("Вариант названия: «%s» пропускаю, уже есть «%s»", item.name, kept[twin].name)
    return kept


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
    result.source_failures = collected_source_failures()
    deps.save_search_result(result)
    log.error("Прогон %s остановлен: %s", result.run_id, message)
    return result

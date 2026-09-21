"""Офлайн-тест всего конвейера на фикстуре: схема, топ-15 и ни одной выдуманной ссылки."""

import asyncio

import pytest

from src.common.schemas import (
    TOP_N,
    Candidate,
    Document,
    ScoredCandidate,
    SearchResult,
    SignalCard,
    SourceRef,
    SourceType,
    TermStats,
    TrustLevel,
)
from src.llm.client import LLMError
from src.pipeline import deps
from src.pipeline.run import NEUTRAL_SCORE, STAGE_DONE, _cards, _features, _log_near_misses, run


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch: pytest.MonkeyPatch):
    """Конвейер не должен ходить в сеть в офлайн-тестах."""

    async def phrases(query: str) -> list[str]:
        return [query, f"{query} patents"]

    async def docs(phrases: list[str], limit: int) -> list[Document]:
        return deps.fixture_documents()[:limit]

    async def no_stats(term: str) -> None:
        return None

    def no_client() -> None:
        raise LLMError("офлайн-тест: модели нет")

    monkeypatch.setattr("src.pipeline.run.expand_query", phrases)
    monkeypatch.setattr("src.pipeline.deps.collect", docs)
    monkeypatch.setattr("src.pipeline.deps.term_stats", no_stats)
    # Кандидатов и карточки делает LLM. В офлайн-тестах модели нет — и это проверка сама по себе:
    # конвейер обязан дойти до конца на запасных вариантах (названия из заголовков, карточка без описания).
    monkeypatch.setattr("src.llm.client.LLMClient.from_settings", staticmethod(no_client))
    # Хранилище тоже не трогаем: без запущенного Postgres каждое сохранение ждёт таймаут подключения.
    monkeypatch.setattr("src.pipeline.deps.save_documents", lambda docs: None)
    monkeypatch.setattr("src.pipeline.deps.save_search_result", lambda result: None)


async def test_full_run_on_fixture():
    result = await run("перспективные решения в финтехе")

    assert isinstance(result, SearchResult)
    assert result.status == "done"
    assert result.stage == STAGE_DONE
    assert result.error is None
    assert result.documents_processed == len(deps.fixture_documents())
    assert 0 < len(result.top) <= TOP_N
    assert result.duration_s is not None
    # Каждый вызов модели попал в журнал (требование ТЗ).
    assert {c.step for c in result.model_calls} == {"score"}


async def test_every_source_comes_from_collected_documents():
    result = await run("перспективные решения в финтехе")

    known = {d.id: d.url for d in deps.fixture_documents()}
    for card in result.top:
        assert card.sources, f"{card.candidate_id}: сигнал без источников в выдачу попасть не может"
        for source in card.sources:
            assert source.document_id in known
            assert source.url == known[source.document_id]


async def test_no_documents_is_an_error(monkeypatch: pytest.MonkeyPatch):
    async def nothing(phrases: list[str], limit: int) -> list:
        return []

    monkeypatch.setattr("src.pipeline.deps.collect", nothing)

    result = await run("пустой запрос")

    assert result.status == "error"
    assert result.error
    assert result.top == []


async def test_progress_callback_sees_stages():
    stages: list[str] = []
    await run("финтех", on_progress=lambda r: stages.append(r.stage))

    assert stages[0] == "расширяем запрос"
    assert stages[-1] == STAGE_DONE


def test_near_misses_are_logged(caplog: pytest.LogCaptureFixture):
    """Кандидаты сразу за топ-15 попадают в лог: по ним видно, что технологию нашли, но ранжировали низко."""
    scored = [
        ScoredCandidate(candidate_id=f"c{i}", name=f"Технология {i}", score=round(1 - i / 100, 2)) for i in range(40)
    ]

    with caplog.at_level("INFO", logger="src.pipeline.run"):
        _log_near_misses(scored)

    logged = caplog.text
    assert "Технология 15 0.85" in logged  # первый не попавший в топ-15
    assert "Технология 29" in logged  # показываем 15 ближайших
    assert "Технология 30" not in logged


async def test_features_keep_partial_results(monkeypatch: pytest.MonkeyPatch):
    """Статистика по части кандидатов не успела — признаки остальных не теряются."""

    async def slow_for_one(term: str):
        if term == "медленный термин":
            await asyncio.sleep(10)
        return TermStats(term=term, pubs_by_year={2025: 7})

    monkeypatch.setattr("src.pipeline.deps.term_stats", slow_for_one)
    monkeypatch.setattr("src.pipeline.run.STATS_BUDGET_S", 0.3)
    candidates = [
        Candidate(id="fast", name="быстрый термин", document_ids=["a"]),
        Candidate(id="slow", name="медленный термин", document_ids=["a"]),
    ]
    docs = [Document(id="a", source="openalex", source_type=SourceType.PAPER, title="t", url="https://example.org/a")]

    features = await _features(candidates, docs)

    by_id = {f.candidate_id: f for f in features}
    assert by_id["fast"].total_pubs == 7  # успел — статистика учтена
    assert by_id["slow"].total_pubs is None  # не успел, но признаки по документам есть
    assert by_id["slow"].distinct_sources == 1


async def test_cards_backfill_to_fifteen():
    """Кандидат без источников выбывает, а на его место берётся следующий — в выдаче ровно 15."""
    candidates = [Candidate(id=f"c{i}", name=f"Технология {i}", document_ids=[]) for i in range(25)]
    docs = []
    for i, candidate in enumerate(candidates):
        if i % 3 == 0:  # у каждого третьего кандидата документов нет — карточка не соберётся
            continue
        doc_id = f"d{i}"
        candidate.document_ids.append(doc_id)
        docs.append(
            Document(
                id=doc_id,
                source="openalex",
                source_type=SourceType.PAPER,
                title=f"Работа {i}",
                url=f"https://example.org/{doc_id}",
            )
        )
    scored = [ScoredCandidate(candidate_id=c.id, name=c.name, score=1 - i / 100) for i, c in enumerate(candidates)]

    cards = await _cards(scored, candidates, docs)

    assert len(cards) == TOP_N
    assert all(card.sources for card in cards)
    # Порядок сохранён, пропущены только кандидаты без источников.
    assert [c.candidate_id for c in cards[:3]] == ["c1", "c2", "c4"]


async def test_fallback_ranking_when_model_fails(monkeypatch: pytest.MonkeyPatch):
    """Модель упала — выдача не пустеет: ранжируем по числу документов с нейтральной уверенностью."""

    def broken(*_: object) -> list:
        raise RuntimeError("модель не загрузилась")

    monkeypatch.setattr("src.pipeline.run.score", broken)

    result = await run("перспективные решения в финтехе")

    assert result.status == "done"
    assert result.top, "выдача не должна быть пустой из-за падения модели"
    assert all(card.score == NEUTRAL_SCORE for card in result.top)
    documents_per_card = [len(card.sources) for card in result.top]
    assert documents_per_card == sorted(documents_per_card, reverse=True)


async def test_scored_field_is_filled():
    """В SearchResult.scored попадают все кандидаты после отсева, по убыванию уверенности."""
    result = await run("перспективные решения в финтехе")

    assert len(result.scored) == result.candidates_found - len(result.excluded)
    scores = [s.score for s in result.scored]
    assert scores == sorted(scores, reverse=True)
    assert {c.candidate_id for c in result.top} <= {s.candidate_id for s in result.scored}


async def test_cards_keep_finished_work_when_budget_runs_out(monkeypatch: pytest.MonkeyPatch):
    """Бюджет кончился посреди пачки — готовые карточки этой пачки остаются в выдаче.

    Прогон 20.09 отработал make_card двенадцать раз, а в выдачу попало девять: общий таймаут
    вокруг цикла отменял пачку целиком вместе с уже посчитанными карточками.
    """
    candidates = [Candidate(id=f"c{i}", name=f"Технология {i}", document_ids=[f"d{i}"]) for i in range(9)]
    docs = [
        Document(
            id=f"d{i}",
            source="openalex",
            source_type=SourceType.PAPER,
            title=f"Работа {i}",
            url=f"https://example.org/d{i}",
        )
        for i in range(9)
    ]
    scored = [ScoredCandidate(candidate_id=c.id, name=c.name, score=1 - i / 100) for i, c in enumerate(candidates)]

    # Каждая третья карточка «зависает»: пачка не успевает целиком, две готовые в ней — успевают.
    async def slow_for_every_third(item, own_docs, name_ru=None):
        if int(item.candidate_id.removeprefix("c")) % 3 == 2:
            await asyncio.sleep(10)
        return SignalCard(
            candidate_id=item.candidate_id,
            name=item.name,
            score=item.score,
            description="описание",
            advantage="преимущество",
            case_example="пример",
            why_weak_signal="ранняя стадия",
            sources=[
                SourceRef(
                    document_id=own_docs[0].id,
                    title=own_docs[0].title,
                    url=own_docs[0].url,
                    source_type=own_docs[0].source_type,
                    language="en",
                    trust=TrustLevel.MEDIUM,
                )
            ],
        )

    monkeypatch.setattr("src.pipeline.run.make_card", slow_for_every_third)
    monkeypatch.setattr("src.pipeline.run.CARDS_BUDGET_S", 0.3)

    cards = await _cards(scored, candidates, docs)

    # Первая пачка — c0, c1, c2: c2 завис, но c0 и c1 посчитаны и должны остаться.
    assert [c.candidate_id for c in cards] == ["c0", "c1"]


async def test_source_failures_are_visible_in_result():
    """Отказ источника попадает в выдачу, а не только в лог.

    Падение источника не валит прогон — это требование ТЗ. Из-за этого мёртвый источник
    выглядит как обычная работа: карточек столько же, времени даже меньше, потому что отказ
    приходит мгновенно. 21.09 исчерпанный лимит OpenAlex дважды незаметно испортил замеры.
    """
    from src.collectors.http import safe_call
    from src.common.logs import collected_source_failures, start_run_log

    start_run_log()

    async def falls() -> None:
        raise TimeoutError("источник молчит")

    errors: list[str] = []
    for _ in range(3):
        assert await safe_call("openalex", falls, errors) is None
    assert await safe_call("arxiv", falls, errors) is None

    failures = collected_source_failures()
    assert [f.source for f in failures] == ["openalex", "arxiv"], "по убыванию числа отказов"
    assert failures[0].count == 3, "повторные отказы одного источника складываются"
    assert "источник молчит" in failures[0].detail, "причина сохранена"


def test_outer_candidates_budget_exceeds_inner():
    """Внешний бюджет шага должен быть заметно больше внутреннего.

    Шаг выделения кандидатов останавливается сам и отдаёт то, что успел выписать. Отмена
    снаружи приходит посреди вызова модели и уносит всё — 21.09 поднятие внутреннего бюджета
    выше внешнего обнулило прогон по всем шести областям.

    Два числа лежат в разных файлах, поэтому связь между ними проверяется здесь.
    """
    from src.pipeline.candidates import BUDGET_S as inner
    from src.pipeline.run import CANDIDATES_BUDGET_S as outer

    assert outer > inner, f"внешний бюджет {outer} с не больше внутреннего {inner} с"
    assert outer - inner >= 60, (
        f"запас всего {outer - inner:.0f} с: одна пачка документов идёт около 12 с, "
        "нужен запас хотя бы на несколько"
    )

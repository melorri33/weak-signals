"""Офлайн-тест всего конвейера на фикстуре: схема, топ-15 и ни одной выдуманной ссылки."""

import asyncio

import pytest

from src.common.schemas import TOP_N, Candidate, Document, ScoredCandidate, SearchResult, SourceType, TermStats
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

    monkeypatch.setattr("src.pipeline.run.expand_query", phrases)
    monkeypatch.setattr("src.pipeline.deps.collect", docs)
    monkeypatch.setattr("src.pipeline.deps.term_stats", no_stats)
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

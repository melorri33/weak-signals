"""Офлайн-тест всего конвейера на фикстуре: схема, топ-15 и ни одной выдуманной ссылки."""

import pytest

from src.common.schemas import TOP_N, ScoredCandidate, SearchResult
from src.pipeline import deps
from src.pipeline.run import STAGE_DONE, _log_near_misses, run


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch: pytest.MonkeyPatch):
    """Конвейер не должен ходить в сеть в офлайн-тестах."""

    async def phrases(query: str) -> list[str]:
        return [query, f"{query} patents"]

    monkeypatch.setattr("src.pipeline.run.expand_query", phrases)


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

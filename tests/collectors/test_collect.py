"""collect(phrases, limit): объединяет источники, дедуплицирует, укладывается в бюджет времени.

OpenAlex и arXiv здесь подменены (см. tests/collectors/test_openalex.py и test_arxiv.py —
там же разбор настоящих ответов на фикстурах). Здесь проверяется только оркестрация.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from src.collectors import arxiv, openalex
from src.common.config import Settings
from src.common.schemas import Document, SourceType

# collect.py и src/collectors/__init__.py оба называют своё имя `collect` — обычный `import src.collectors.collect`
# ходит по атрибутам пакета и вместо модуля вернёт переэкспортированную функцию. import_module — в обход.
collect_module = importlib.import_module("src.collectors.collect")


def _doc(doc_id: str, title: str | None = None) -> Document:
    # Название по умолчанию уникально — dedupe() без DOI склеивает по нормализованному заголовку.
    return Document(
        id=doc_id,
        source=doc_id.split(":")[0],
        source_type=SourceType.PAPER,
        title=title or f"документ {doc_id}",
        url=f"https://example.org/{doc_id}",
    )


async def test_collect_combines_both_sources(monkeypatch: pytest.MonkeyPatch):
    async def fake_openalex(phrase, settings, client, limit=200):
        return [_doc("openalex:1")]

    async def fake_arxiv(phrase, settings, client, limit=50):
        return [_doc("arxiv:1")]

    monkeypatch.setattr(openalex, "search", fake_openalex)
    monkeypatch.setattr(arxiv, "search", fake_arxiv)

    docs = await collect_module.collect(["quantum sensing"])

    assert {d.id for d in docs} == {"openalex:1", "arxiv:1"}


async def test_collect_empty_phrases_returns_empty():
    assert await collect_module.collect([]) == []


async def test_collect_dedupes_across_phrases(monkeypatch: pytest.MonkeyPatch):
    async def fake_openalex(phrase, settings, client, limit=200):
        return [_doc("openalex:1", title="один и тот же документ")]

    async def fake_arxiv(phrase, settings, client, limit=50):
        return []

    monkeypatch.setattr(openalex, "search", fake_openalex)
    monkeypatch.setattr(arxiv, "search", fake_arxiv)

    docs = await collect_module.collect(["фраза 1", "фраза 2"])

    assert len(docs) == 1  # найден по двум фразам, но это один документ


async def test_collect_survives_one_source_failing(monkeypatch: pytest.MonkeyPatch):
    async def broken(phrase, settings, client, limit=200):
        raise RuntimeError("источник упал")

    async def fake_arxiv(phrase, settings, client, limit=50):
        return [_doc("arxiv:1")]

    monkeypatch.setattr(openalex, "search", broken)
    monkeypatch.setattr(arxiv, "search", fake_arxiv)

    docs = await collect_module.collect(["quantum sensing"])

    assert [d.id for d in docs] == ["arxiv:1"]


async def test_collect_respects_limit(monkeypatch: pytest.MonkeyPatch):
    async def fake_openalex(phrase, settings, client, limit=200):
        return [_doc(f"openalex:{i}") for i in range(10)]

    async def fake_arxiv(phrase, settings, client, limit=50):
        return []

    monkeypatch.setattr(openalex, "search", fake_openalex)
    monkeypatch.setattr(arxiv, "search", fake_arxiv)

    docs = await collect_module.collect(["quantum sensing"], limit=3)

    assert len(docs) == 3


async def test_collect_drops_phrases_that_exceed_budget(monkeypatch: pytest.MonkeyPatch):
    async def slow_openalex(phrase, settings, client, limit=200):
        await asyncio.sleep(1.0)
        return [_doc("openalex:slow")]

    async def fake_arxiv(phrase, settings, client, limit=50):
        return []

    monkeypatch.setattr(openalex, "search", slow_openalex)
    monkeypatch.setattr(arxiv, "search", fake_arxiv)
    monkeypatch.setattr(collect_module, "get_settings", lambda: Settings(collect_budget_s=0.1))

    docs = await collect_module.collect(["quantum sensing"])

    assert docs == []  # не успели — отдаём то, что есть (пусто), а не падаем


@pytest.mark.network
async def test_collect_quantum_sensing_returns_50_plus_real_documents():
    """Критерий готовности дня 1-2 из CLAUDE.md / .claude/agents/data-engineer.md.

    OpenAlex и arXiv в теме «quantum sensing» сильно пересекаются — многие препринты имеют
    опубликованную версию с тем же названием, dedupe() законно склеивает их в один документ.
    Поэтому здесь проверяется только итоговое число, а не то, что оба источника выжили.
    """
    docs = await collect_module.collect(["quantum sensing"], limit=200)

    assert len(docs) >= 50

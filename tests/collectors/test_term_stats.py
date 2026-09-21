"""term_stats(term): источники подменены — оркестрация, кэш и обработка ошибок проверяются реально.

Разбор настоящих ответов каждого источника — в test_openalex.py, test_hackernews.py, test_wikipedia.py.
"""

from __future__ import annotations

import asyncio
import importlib
import time

import pytest

from src.collectors import hackernews, openalex, wikipedia
from src.common.schemas import TermStats

# term_stats.py и src/collectors/__init__.py оба называют своё имя `term_stats` — обычный import
# ходит по атрибутам пакета и вместо модуля вернёт переэкспортированную функцию. import_module — в обход.
term_stats_module = importlib.import_module("src.collectors.term_stats")


def _patch_all(
    monkeypatch: pytest.MonkeyPatch,
    *,
    years=None,
    types=None,
    orgs=7,
    news=None,
    wiki_en=True,
    wiki_ru=False,
):
    years = years if years is not None else {2024: 5, 2025: 10}
    types = types if types is not None else {"article": 10}
    news = news if news is not None else {2025: 3}
    calls = {"n": 0}

    async def fake_years(term, settings, client):
        calls["n"] += 1
        return years

    async def fake_types(term, settings, client):
        return types

    async def fake_orgs(term, settings, client):
        return orgs

    async def fake_news(term, settings, client):
        return news

    async def fake_wiki(term, lang, settings, client):
        return wiki_en if lang == "en" else wiki_ru

    monkeypatch.setattr(openalex, "year_counts", fake_years)
    monkeypatch.setattr(openalex, "type_counts", fake_types)
    monkeypatch.setattr(hackernews, "news_by_year", fake_news)
    monkeypatch.setattr(wikipedia, "has_article", fake_wiki)
    return calls


async def test_term_stats_fills_all_available_fields(monkeypatch: pytest.MonkeyPatch):
    _patch_all(monkeypatch)

    stats = await term_stats_module.term_stats("solid-state battery")

    assert isinstance(stats, TermStats)
    assert stats.term == "solid-state battery"
    assert stats.pubs_by_year == {2024: 5, 2025: 10}
    assert stats.pubs_by_type == {"article": 10}
    # Число организаций больше не запрашиваем: третий запрос к OpenAlex на кандидата,
    # а поле не читает ни модель, ни правила отсева. См. docstring term_stats.
    assert stats.distinct_orgs is None
    assert stats.news_by_year == {2025: 3}
    assert stats.wikipedia_en is True
    assert stats.wikipedia_ru is False
    # Патенты и упоминания стандартов — источника ещё нет.
    assert stats.patents_by_year is None
    assert stats.standard_mentions is None
    assert stats.errors == []


async def test_term_stats_records_failed_source_as_none_with_error(monkeypatch: pytest.MonkeyPatch):
    _patch_all(monkeypatch)

    async def broken(term, settings, client):
        raise RuntimeError("openalex недоступен")

    monkeypatch.setattr(openalex, "year_counts", broken)

    stats = await term_stats_module.term_stats("solid-state battery")

    assert stats.pubs_by_year == {}  # None по контракту превращается в {}, не в 0
    assert "openalex_years" in stats.errors
    assert stats.pubs_by_type == {"article": 10}  # падение одного источника не задевает остальные


async def test_term_stats_uses_cache_on_second_call(monkeypatch: pytest.MonkeyPatch):
    calls = _patch_all(monkeypatch)

    await term_stats_module.term_stats("solid-state battery")
    await term_stats_module.term_stats("solid-state battery")

    assert calls["n"] == 1  # второй вызов пришёл из кэша


async def test_term_stats_int_keys_survive_cache_roundtrip(monkeypatch: pytest.MonkeyPatch):
    _patch_all(monkeypatch)

    await term_stats_module.term_stats("solid-state battery")
    stats = await term_stats_module.term_stats("solid-state battery")  # из кэша (JSON теряет типы ключей)

    assert all(isinstance(y, int) for y in stats.pubs_by_year)
    assert all(isinstance(y, int) for y in stats.news_by_year)


async def test_term_stats_queries_sources_in_parallel(monkeypatch: pytest.MonkeyPatch):
    """OpenAlex/HN/Wikipedia не должны ждать друг друга — иначе term_stats для нескольких
    кандидатов параллельно (STATS_CONCURRENCY в пайплайне) не укладывается в бюджет времени."""
    delay = 0.05
    started: list[float] = []

    async def slow(value):
        started.append(time.monotonic())
        await asyncio.sleep(delay)
        return value

    monkeypatch.setattr(openalex, "year_counts", lambda term, settings, client: slow({2025: 1}))
    monkeypatch.setattr(openalex, "type_counts", lambda term, settings, client: slow({"article": 1}))
    monkeypatch.setattr(hackernews, "news_by_year", lambda term, settings, client: slow({2025: 1}))
    monkeypatch.setattr(wikipedia, "has_article", lambda term, lang, settings, client: slow(True))

    await term_stats_module.term_stats("solid-state battery")

    # Не сравниваем с общим временем вызова: создание httpx.AsyncClient в этом окружении само
    # по себе занимает ~0.5 с независимо от источников — а вот разброс между стартами источников
    # это не маскирует и достоверно показывает, ждут они друг друга или нет.
    assert len(started) == 5  # годы, типы, Hacker News и две Википедии
    assert max(started) - min(started) < delay  # все источники стартовали почти одновременно


@pytest.mark.network
async def test_term_stats_solid_state_battery_matches_spec():
    """Критерий готовности из спецификации term_stats: сотни работ за 2024-2025, 5 ключей новостей, есть в вики."""
    stats = await term_stats_module.term_stats("solid-state battery")

    assert stats.pubs_by_year.get(2024, 0) > 100 or stats.pubs_by_year.get(2025, 0) > 100
    assert stats.news_by_year is not None and set(stats.news_by_year) == {2016, 2023, 2024, 2025, 2026}
    assert stats.wikipedia_en is True

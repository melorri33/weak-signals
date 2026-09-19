"""term_stats(term): источники подменены — оркестрация, кэш и обработка ошибок проверяются реально.

Разбор настоящих ответов каждого источника — в test_openalex.py, test_hackernews.py, test_wikipedia.py.
"""

from __future__ import annotations

import importlib

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
    monkeypatch.setattr(openalex, "org_count", fake_orgs)
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
    assert stats.distinct_orgs == 7
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
    assert stats.distinct_orgs == 7  # падение одного источника не задевает остальные


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


@pytest.mark.network
async def test_term_stats_solid_state_battery_matches_spec():
    """Критерий готовности из спецификации term_stats: сотни работ за 2024-2025, 5 ключей новостей, есть в вики."""
    stats = await term_stats_module.term_stats("solid-state battery")

    assert stats.pubs_by_year.get(2024, 0) > 100 or stats.pubs_by_year.get(2025, 0) > 100
    assert stats.news_by_year is not None and set(stats.news_by_year) == {2016, 2023, 2024, 2025, 2026}
    assert stats.wikipedia_en is True

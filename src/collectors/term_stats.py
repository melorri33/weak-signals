"""term_stats(term): статистика по одному термину-кандидату из открытых источников.

- pubs_by_year, pubs_by_type, distinct_orgs — OpenAlex (поиск точной фразы).
- news_by_year — Hacker News (Algolia); GDELT сейчас отвечает 429, поэтому не используется.
- wikipedia_en / wikipedia_ru — есть ли статья с совпадающим названием.
- patents_by_year, standard_mentions — пока None, источника нет (нужен ключ PatentsView).

Источник не ответил → поле None (не 0/{}) и его имя попадает в TermStats.errors.
Ответы кэшируются (src.collectors.cache), ключ = источник + термин, TTL = CACHE_TTL_DAYS.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from src.collectors import cache, hackernews, openalex, wikipedia
from src.collectors.http import safe_call, user_agent
from src.common.config import get_settings
from src.common.schemas import TermStats

T = TypeVar("T")


def _int_keys(value: dict) -> dict[int, int]:
    return {int(k): v for k, v in value.items()}


async def _cached(
    source: str,
    term: str,
    errors: list[str],
    call: Callable[[], Awaitable[T]],
    normalize: Callable[[T], T] = lambda v: v,
) -> T | None:
    """Значение из кэша (нормализованное — JSON теряет типы ключей), иначе запрос к источнику."""
    if (hit := cache.get(source, term)) is not None:
        return normalize(hit)
    value = await safe_call(source, call, errors)
    if value is None:
        return None
    cache.set(source, term, value=value)
    return normalize(value)


async def term_stats(term: str) -> TermStats:
    """Статистика по термину (collectors.term_stats)."""
    settings = get_settings()
    errors: list[str] = []

    async with httpx.AsyncClient(headers={"User-Agent": user_agent("term stats", settings)}) as client:
        pubs_by_year = await _cached(
            "openalex_years", term, errors, lambda: openalex.year_counts(term, settings, client), _int_keys
        )
        pubs_by_type = await _cached(
            "openalex_types", term, errors, lambda: openalex.type_counts(term, settings, client)
        )
        distinct_orgs = await _cached("openalex_orgs", term, errors, lambda: openalex.org_count(term, settings, client))
        news_by_year = await _cached(
            "hn_news", term, errors, lambda: hackernews.news_by_year(term, settings, client), _int_keys
        )
        wikipedia_en = await _cached(
            "wikipedia_en", term, errors, lambda: wikipedia.has_article(term, "en", settings, client)
        )
        wikipedia_ru = await _cached(
            "wikipedia_ru", term, errors, lambda: wikipedia.has_article(term, "ru", settings, client)
        )

    return TermStats(
        term=term,
        pubs_by_year=pubs_by_year or {},
        pubs_by_type=pubs_by_type,
        patents_by_year=None,
        news_by_year=news_by_year,
        wikipedia_ru=wikipedia_ru,
        wikipedia_en=wikipedia_en,
        standard_mentions=None,
        distinct_orgs=distinct_orgs,
        errors=errors,
    )

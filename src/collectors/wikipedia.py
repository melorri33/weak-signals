"""Википедия: есть ли статья о термине (признак зрелости технологии).

Без контакта в User-Agent отвечает 403.
"""

from __future__ import annotations

import re

import httpx

from src.collectors.http import user_agent
from src.common.config import Settings

_API = "https://{lang}.wikipedia.org/w/api.php"
_MATCH_THRESHOLD = 0.6


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _matches(term: str, title: str) -> bool:
    term_tokens, title_tokens = _tokens(term), _tokens(title)
    if not term_tokens or not title_tokens:
        return False
    overlap = len(term_tokens & title_tokens) / len(term_tokens | title_tokens)
    return overlap >= _MATCH_THRESHOLD


async def has_article(term: str, lang: str, settings: Settings, client: httpx.AsyncClient) -> bool:
    """True, если среди первых 5 результатов поиска есть статья, чьё название совпадает с термином."""
    r = await client.get(
        _API.format(lang=lang),
        params={"action": "query", "list": "search", "srsearch": f'"{term}"', "srlimit": 5, "format": "json"},
        headers={"User-Agent": user_agent("wikipedia probe", settings)},
        timeout=settings.source_timeout_s,
    )
    r.raise_for_status()
    hits = r.json()["query"]["search"]
    return any(_matches(term, hit["title"]) for hit in hits)

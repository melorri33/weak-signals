"""Hacker News (Algolia): число историй с точной фразой — заменяет GDELT (тот отвечает 429).

Строго последовательные запросы с паузой — при параллельных Algolia отвечает 403 и блокирует.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx

from src.common.config import Settings

API = "https://hn.algolia.com/api/v1/search"
PAUSE_S = 0.3
# 2016–2022 одним запросом (ключ 2016), дальше по годам — 5 запросов вместо 11.
_BUCKETS = [(2016, 2023), (2023, 2024), (2024, 2025), (2025, 2026), (2026, 2027)]


def _year_ts(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=UTC).timestamp())


async def news_by_year(term: str, settings: Settings, client: httpx.AsyncClient) -> dict[int, int]:
    """Число историй HN с точной фразой по годам (раньше 2016 не считаем)."""
    out: dict[int, int] = {}
    for i, (start, end) in enumerate(_BUCKETS):
        if i > 0:
            await asyncio.sleep(PAUSE_S)
        numeric_filter = f"created_at_i>={_year_ts(start)},created_at_i<{_year_ts(end)}"
        r = await client.get(
            API,
            params={"query": f'"{term}"', "tags": "story", "hitsPerPage": 0, "numericFilters": numeric_filter},
            timeout=settings.source_timeout_s,
        )
        r.raise_for_status()
        out[start] = r.json()["nbHits"]
    return out

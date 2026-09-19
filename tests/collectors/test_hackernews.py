"""Hacker News (Algolia): пять последовательных запросов по годам — без сети."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from src.collectors import hackernews
from src.collectors.http import RateLimiter
from tests.collectors.conftest import raw_fixture


async def test_news_by_year_makes_five_sequential_requests(settings):
    seen_filters: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_filters.append(request.url.params["numericFilters"])
        body = json.loads(raw_fixture("hn_solid_state_battery_2025.json"))
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        years = await hackernews.news_by_year("solid-state battery", settings, client)

    assert list(years) == [2016, 2023, 2024, 2025, 2026]
    assert len(seen_filters) == 5


async def test_news_by_year_uses_exact_phrase(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.setdefault("queries", []).append(request.url.params["query"])
        return httpx.Response(200, json={"nbHits": 0})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await hackernews.news_by_year("quantum sensing", settings, client)

    assert all(q == '"quantum sensing"' for q in seen["queries"])


async def test_news_by_year_reads_nbhits(settings):
    body = json.loads(raw_fixture("hn_solid_state_battery_2025.json"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        years = await hackernews.news_by_year("solid-state battery", settings, client)

    assert years[2025] == body["nbHits"]


async def test_concurrent_calls_do_not_race_requests(settings, monkeypatch: pytest.MonkeyPatch):
    """Конвейер вызывает term_stats для нескольких кандидатов параллельно — HN не должен получать
    от них одновременные запросы (иначе Algolia отвечает 403), лимитер должен быть общим на процесс."""
    interval_s = 0.05
    monkeypatch.setattr(hackernews, "_rate_limiter", RateLimiter(interval_s=interval_s))
    timestamps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timestamps.append(time.monotonic())
        return httpx.Response(200, json={"nbHits": 0})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await asyncio.gather(
            hackernews.news_by_year("solid-state battery", settings, client),
            hackernews.news_by_year("quantum sensing", settings, client),
        )

    assert len(timestamps) == 10  # 5 запросов на термин
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)]
    assert all(gap >= interval_s * 0.9 for gap in gaps)  # 10% допуск на точность таймера

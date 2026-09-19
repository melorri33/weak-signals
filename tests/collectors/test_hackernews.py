"""Hacker News (Algolia): пять последовательных запросов по годам — без сети."""

from __future__ import annotations

import json

import httpx

from src.collectors import hackernews
from tests.collectors.conftest import raw_fixture


async def test_news_by_year_makes_five_sequential_requests(settings):
    seen_filters: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_filters.append(request.url.params["numericFilters"])
        body = json.loads(raw_fixture("hn_machine_unlearning_2025.json"))
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        years = await hackernews.news_by_year("machine unlearning", settings, client)

    assert list(years) == [2016, 2023, 2024, 2025, 2026]
    assert len(seen_filters) == 5


async def test_news_by_year_uses_exact_phrase(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.setdefault("queries", []).append(request.url.params["query"])
        return httpx.Response(200, json={"nbHits": 0})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await hackernews.news_by_year("space data centers", settings, client)

    assert all(q == '"space data centers"' for q in seen["queries"])


async def test_news_by_year_reads_nbhits(settings):
    body = json.loads(raw_fixture("hn_machine_unlearning_2025.json"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        years = await hackernews.news_by_year("machine unlearning", settings, client)

    assert years[2025] == body["nbHits"]

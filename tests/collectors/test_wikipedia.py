"""Википедия: совпадение статьи по токенам (Жаккар ≥ 0.6) — на сохранённых ответах, без сети."""

from __future__ import annotations

import json

import httpx

from src.collectors import wikipedia
from tests.collectors.conftest import raw_fixture


def _mock_client(body: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_has_article_true_when_title_matches(settings):
    body = json.loads(raw_fixture("wikipedia_en_machine_unlearning.json"))
    async with _mock_client(body) as client:
        found = await wikipedia.has_article("machine unlearning", "en", settings, client)

    assert found is True


async def test_has_article_false_when_no_hits(settings):
    body = json.loads(raw_fixture("wikipedia_en_no_match.json"))
    async with _mock_client(body) as client:
        found = await wikipedia.has_article("zzqx nonexistent weak signal term", "en", settings, client)

    assert found is False


async def test_has_article_requires_high_overlap(settings):
    """Совпадение по одному слову из трёх не считается — порог Жаккара 0.6."""
    body = {"query": {"search": [{"title": "Machine learning"}]}}
    async with _mock_client(body) as client:
        found = await wikipedia.has_article("federated credit scoring", "en", settings, client)

    assert found is False


async def test_has_article_sends_contact_in_user_agent(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["user_agent"] = request.headers["User-Agent"]
        return httpx.Response(200, json={"query": {"search": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await wikipedia.has_article("term", "en", settings, client)

    assert "test@example.org" in seen["user_agent"]

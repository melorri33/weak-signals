"""Разбор ответов OpenAlex на сохранённых фикстурах — без сети. Числа сверены живым запросом 19.09.2026."""

from __future__ import annotations

import httpx
import pytest

from src.collectors import openalex
from tests.collectors.conftest import raw_fixture


def _mock_client(body: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_search_parses_documents_with_required_fields(settings):
    body = raw_fixture("openalex_works_quantum_sensing.json")
    async with _mock_client(body) as client:
        docs = await openalex.search("quantum sensing", settings, client)

    assert len(docs) == 5
    for doc in docs:
        assert doc.source == "openalex"
        assert doc.id.startswith("openalex:")
        assert doc.title
        assert doc.url
        assert doc.language
        assert doc.found_by == "quantum sensing"


async def test_search_rebuilds_abstract_from_inverted_index(settings):
    body = raw_fixture("openalex_works_quantum_sensing.json")
    async with _mock_client(body) as client:
        docs = await openalex.search("quantum sensing", settings, client)

    with_abstract = next(d for d in docs if d.title == "Quantum sensing")
    assert with_abstract.abstract is not None
    assert with_abstract.abstract.startswith("Quantum technologies are increasingly driving")


async def test_search_collects_authors_and_organizations(settings):
    body = raw_fixture("openalex_works_quantum_sensing.json")
    async with _mock_client(body) as client:
        docs = await openalex.search("quantum sensing", settings, client)

    doc = next(d for d in docs if d.title == "Quantum sensing")
    assert "Christian L. Degen" in doc.authors
    assert "ETH Zurich" in doc.organizations
    assert len(doc.organizations) == len(set(doc.organizations))  # без дублей


async def test_year_counts_only_digit_keys(settings):
    body = raw_fixture("openalex_years_machine_unlearning.json")
    async with _mock_client(body) as client:
        years = await openalex.year_counts("machine unlearning", settings, client)

    assert years
    assert all(isinstance(y, int) for y in years)
    assert years[2025] > 100  # сотни работ в 2024-2025 — как в спецификации


async def test_type_counts_has_article_or_preprint(settings):
    body = raw_fixture("openalex_types_machine_unlearning.json")
    async with _mock_client(body) as client:
        types = await openalex.type_counts("machine unlearning", settings, client)

    assert types
    assert set(types) & {"article", "preprint"}


async def test_org_count_reads_groups_count(settings):
    body = raw_fixture("openalex_orgs_machine_unlearning.json")
    async with _mock_client(body) as client:
        n = await openalex.org_count("machine unlearning", settings, client)

    assert n > 0


async def test_search_uses_exact_phrase_filter(settings):
    """Без кавычек OpenAlex ищет слова по отдельности — фраза обязана попасть в filter в кавычках."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["filter"] = request.url.params["filter"]
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await openalex.search("space data centers", settings, client)

    assert seen["filter"] == 'title_and_abstract.search:"space data centers"'


@pytest.mark.network
async def test_search_matches_live_api():
    """Живой запрос — только для ручной сверки чисел, в общий прогон офлайн-тестов не входит."""
    from src.common.config import get_settings

    async with httpx.AsyncClient() as client:
        years = await openalex.year_counts("quantum sensing", get_settings(), client)

    assert years
    assert sum(years.values()) > 1000

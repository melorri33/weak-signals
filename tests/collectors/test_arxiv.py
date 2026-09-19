"""Разбор Atom-ответа arXiv на сохранённой фикстуре — без сети."""

from __future__ import annotations

import httpx

from src.collectors import arxiv
from tests.collectors.conftest import raw_fixture


def _mock_client(body: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_search_parses_entries(settings):
    body = raw_fixture("arxiv_quantum_sensing.xml")
    async with _mock_client(body) as client:
        docs = await arxiv.search("quantum sensing", settings, client)

    assert len(docs) == 5
    for doc in docs:
        assert doc.source == "arxiv"
        assert doc.source_type == "preprint"
        assert doc.id.startswith("arxiv:")
        assert doc.title
        assert doc.url.startswith("http")
        assert doc.found_by == "quantum sensing"


async def test_search_collects_authors_and_abstract(settings):
    body = raw_fixture("arxiv_quantum_sensing.xml")
    async with _mock_client(body) as client:
        docs = await arxiv.search("quantum sensing", settings, client)

    assert all(doc.authors for doc in docs)
    assert all(doc.abstract for doc in docs)
    # Заголовок и аннотация не должны содержать переносы строк из Atom-разметки.
    assert all("\n" not in doc.title for doc in docs)


async def test_search_quotes_exact_phrase(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["search_query"] = request.url.params["search_query"]
        return httpx.Response(200, content=b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await arxiv.search("solid-state battery", settings, client)

    assert seen["search_query"] == 'all:"solid-state battery"'

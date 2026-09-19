"""arXiv: свежие препринты. Atom XML, не чаще одного запроса в 3 секунды (условие API)."""

from __future__ import annotations

from datetime import datetime
from xml.etree import ElementTree

import httpx

from src.collectors.http import RateLimiter
from src.common.config import Settings
from src.common.schemas import Document, SourceType

API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# arXiv просит не чаще 1 запроса в 3 секунды — общий на процесс, источник один для всех фраз.
_rate_limiter = RateLimiter(interval_s=3.0)


def _text(entry: ElementTree.Element, tag: str) -> str | None:
    node = entry.find(f"{_ATOM}{tag}")
    return node.text.strip() if node is not None and node.text else None


def _to_document(entry: ElementTree.Element, phrase: str) -> Document:
    entry_id = _text(entry, "id") or ""
    authors = [
        name.text.strip()
        for author in entry.findall(f"{_ATOM}author")
        if (name := author.find(f"{_ATOM}name")) is not None and name.text
    ]
    published = None
    if published_raw := _text(entry, "published"):
        try:
            published = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).date()
        except ValueError:
            published = None
    summary = _text(entry, "summary")
    return Document(
        id=f"arxiv:{entry_id.rsplit('/', 1)[-1]}",
        source="arxiv",
        source_type=SourceType.PREPRINT,
        title=" ".join((_text(entry, "title") or "").split()),
        abstract=" ".join(summary.split()) if summary else None,
        url=entry_id,
        published=published,
        language="en",
        authors=authors,
        found_by=phrase,
    )


async def search(phrase: str, settings: Settings, client: httpx.AsyncClient, limit: int = 50) -> list[Document]:
    """Препринты, где фраза встречается в заголовке, аннотации или тексте."""
    await _rate_limiter.wait()
    params = {
        "search_query": f'all:"{phrase}"',
        "start": "0",
        "max_results": str(limit),
        "sortBy": "relevance",
    }
    r = await client.get(API, params=params, timeout=settings.source_timeout_s)
    r.raise_for_status()
    root = ElementTree.fromstring(r.text)
    return [_to_document(entry, phrase) for entry in root.findall(f"{_ATOM}entry")]

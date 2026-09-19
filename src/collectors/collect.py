"""collect(phrases, limit=500): документы по всем источникам сразу, без дублей.

Фразы опрашиваются параллельно (arXiv сам держит паузу в 3 с между запросами через RateLimiter,
поэтому не тормозит остальных). Весь сбор укладывается в COLLECT_BUDGET_S: то, что не успело —
пропускается, уже собранное возвращается.
"""

from __future__ import annotations

import asyncio
from itertools import zip_longest

import httpx

from src.collectors import arxiv, openalex
from src.collectors.dedupe import dedupe
from src.collectors.http import safe_call, user_agent
from src.common.config import Settings, get_settings
from src.common.logs import get_logger
from src.common.schemas import Document

log = get_logger(__name__)


async def collect(phrases: list[str], limit: int = 500) -> list[Document]:
    """Поиск по всем источникам сразу, без дублей (collectors.collect)."""
    if not phrases:
        return []
    settings = get_settings()
    docs: list[Document] = []
    errors: list[str] = []

    async with httpx.AsyncClient(headers={"User-Agent": user_agent("document collector", settings)}) as client:
        tasks = [asyncio.ensure_future(_collect_phrase(phrase, settings, client, errors)) for phrase in phrases]
        done, pending = await asyncio.wait(tasks, timeout=settings.collect_budget_s)
        for task in pending:
            task.cancel()
        if pending:
            log.warning("collect: бюджет времени исчерпан, %d фраз(ы) из %d не успели", len(pending), len(phrases))
        for task in done:
            docs.extend(task.result())

    return dedupe(docs)[:limit]


async def _collect_phrase(
    phrase: str, settings: Settings, client: httpx.AsyncClient, errors: list[str]
) -> list[Document]:
    openalex_docs, arxiv_docs = await asyncio.gather(
        safe_call("openalex", lambda: openalex.search(phrase, settings, client), errors),
        safe_call("arxiv", lambda: arxiv.search(phrase, settings, client), errors),
    )
    return _interleave(openalex_docs or [], arxiv_docs or [])


def _interleave(*groups: list[Document]) -> list[Document]:
    """По одному документу из каждого источника по кругу — иначе при обрезке по limit
    источник с большей выдачей (OpenAlex, per_page=200) вытесняет остальные ещё до дедупликации."""
    return [doc for row in zip_longest(*groups) for doc in row if doc is not None]

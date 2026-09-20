"""collect(phrases, limit=500): документы по всем источникам сразу, без дублей.

Опрашиваем параллельно каждую пару «фраза + источник». Весь сбор укладывается в COLLECT_BUDGET_S:
что не успело — пропускается, уже собранное возвращается.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from itertools import zip_longest

import httpx

from src.collectors import arxiv, news, openalex
from src.collectors.dedupe import dedupe
from src.collectors.http import safe_call, user_agent
from src.common.config import get_settings
from src.common.logs import get_logger
from src.common.schemas import Document

log = get_logger(__name__)


async def collect(phrases: list[str], limit: int = 500) -> list[Document]:
    """Поиск по всем источникам сразу, без дублей (collectors.collect).

    Каждая пара «фраза + источник» — отдельная задача. Это важно: arXiv просит паузу 3 с между
    запросами, и на 18 фразах он один занимает почти минуту. Если бы фраза ждала все свои источники
    разом, исчерпание бюджета отменяло бы вместе с arXiv и уже полученные новости — на живом прогоне
    это давало ноль документов при живых лентах.
    """
    if not phrases:
        return []
    settings = get_settings()
    by_source: dict[str, list[Document]] = {"news": [], "openalex": [], "arxiv": []}
    errors: list[str] = []

    async with httpx.AsyncClient(headers={"User-Agent": user_agent("document collector", settings)}) as client:
        tasks: dict[asyncio.Future[list[Document] | None], str] = {}
        for phrase in phrases:
            tasks[asyncio.ensure_future(news.search(phrase, settings, client, errors))] = "news"
            for source, search in (("openalex", openalex.search), ("arxiv", arxiv.search)):
                call = asyncio.ensure_future(
                    safe_call(source, lambda s=search, p=phrase: s(p, settings, client), errors)
                )
                tasks[call] = source
        done, pending = await asyncio.wait(tasks, timeout=settings.collect_budget_s)
        for task in pending:
            task.cancel()
        if pending:
            missed = Counter(tasks[task] for task in pending)
            log.warning(
                "collect: бюджет %.0f с исчерпан, не успели: %s",
                settings.collect_budget_s,
                ", ".join(f"{source} ×{n}" for source, n in missed.items()),
            )
        for task in done:
            if found := task.result():
                by_source[tasks[task]].extend(found)

    # Новости первыми: 71% источников датасета организаторов — техноновости, и именно в них
    # живут ранние технологии (раунды стартапов, первые внедрения), которых ещё нет в науке.
    docs = _interleave(by_source["news"], by_source["openalex"], by_source["arxiv"])
    return dedupe(docs)[:limit]


def _interleave(*groups: list[Document]) -> list[Document]:
    """По одному документу из каждого источника по кругу — иначе при обрезке по limit
    источник с большей выдачей (OpenAlex, per_page=200) вытесняет остальные ещё до дедупликации."""
    return [doc for row in zip_longest(*groups) for doc in row if doc is not None]

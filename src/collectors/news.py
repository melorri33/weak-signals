"""Новости техномедиа по поисковой фразе: RSS-ленты поиска изданий и Bing News.

Зачем: 71% источников датасета организаторов — техноновости, ещё 16% — блоги и пресс-релизы.
Треть описаний — про свежие раунды стартапов, чего в научных базах нет вообще. Прогон сбора
только по OpenAlex и arXiv находил 14 технологий датасета из 100 — остальные живут в новостях.

Список лент — config/news_feeds.yaml. Ссылки Bing разворачиваем до настоящего адреса издания:
уровень доверия считается по домену, а агрегатор сам по себе основанием для сигнала быть не может.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree

import httpx
import yaml

from src.collectors.http import safe_call
from src.common.config import Settings
from src.common.schemas import Document, SourceType

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "news_feeds.yaml"

# Сколько новостей берём из одной ленты: дальше начинается выдача «по касательной к фразе».
PER_FEED_LIMIT = 20

_TAG_RE = re.compile(r"<[^>]+>")


@lru_cache(maxsize=1)
def feeds() -> list[dict[str, str]]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["search_feeds"]


def _unwrap_bing(url: str) -> str:
    """https://www.bing.com/news/apiclick.aspx?...&url=https%3a%2f%2fexample.com%2fa → https://example.com/a."""
    target = parse_qs(urlparse(url).query).get("url", [])
    return target[0] if target else url


def _domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.").lower()


def _text(item: ElementTree.Element, tag: str) -> str | None:
    node = item.find(tag)
    return node.text.strip() if node is not None and node.text else None


def _published(raw: str | None):
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).date()
    except (TypeError, ValueError):
        return None


def _summary(raw: str | None) -> str | None:
    """В description лежит HTML-анонс — оставляем текст."""
    if not raw:
        return None
    text = " ".join(_TAG_RE.sub(" ", raw).split())
    return text or None


def parse_feed(xml: str, phrase: str, unwrap: str | None = None) -> list[Document]:
    """Разобрать RSS-ленту в документы. Ссылки Bing разворачиваем до адреса издания."""
    root = ElementTree.fromstring(xml)
    docs = []
    for item in root.iter("item"):
        link = _text(item, "link")
        title = _text(item, "title")
        if not link or not title:
            continue
        url = _unwrap_bing(link) if unwrap == "bing" else link
        domain = _domain(url)
        if not domain or domain.endswith("bing.com"):
            continue  # развернуть не удалось: агрегатор основанием для сигнала быть не может
        docs.append(
            Document(
                id=f"{domain}:{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
                source=domain,
                source_type=SourceType.NEWS,
                title=title,
                abstract=_summary(_text(item, "description")),
                url=url,
                published=_published(_text(item, "pubDate")),
                language="en",
                found_by=phrase,
            )
        )
    return docs


async def _from_feed(feed: dict[str, str], phrase: str, settings: Settings, client: httpx.AsyncClient) -> list[str]:
    url = feed["url"].format(query=phrase.replace(" ", "+"))
    r = await client.get(url, timeout=settings.source_timeout_s, follow_redirects=True)
    r.raise_for_status()
    return parse_feed(r.text, phrase, feed.get("unwrap"))[:PER_FEED_LIMIT]


async def search(phrase: str, settings: Settings, client: httpx.AsyncClient, errors: list[str]) -> list[Document]:
    """Новости по фразе из всех лент сразу; упавшая лента не мешает остальным."""
    calls = [
        safe_call(f"news:{feed['name']}", lambda f=feed: _from_feed(f, phrase, settings, client), errors)
        for feed in feeds()
    ]
    results = await asyncio.gather(*calls)
    return [doc for group in results if group for doc in group]

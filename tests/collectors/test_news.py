"""Разбор новостных RSS-лент на сохранённых ответах — без сети."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from src.collectors import news
from tests.collectors.conftest import raw_fixture


def _mock_client(body: str, status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body.encode("utf-8"))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_parses_publisher_feed():
    docs = news.parse_feed(raw_fixture("news_techcrunch_quantum_sensing.xml"), "quantum sensing")

    assert docs
    for doc in docs:
        assert doc.source == "techcrunch.com"
        assert doc.source_type == "news"
        assert doc.url.startswith("https://techcrunch.com/")
        assert doc.found_by == "quantum sensing"
        assert doc.title
    assert any(isinstance(doc.published, date) for doc in docs)
    # В анонсе приходит HTML — в аннотацию он попадать не должен.
    assert all("<" not in (doc.abstract or "") for doc in docs)


def test_bing_links_are_unwrapped_to_the_publisher():
    """Ссылка агрегатора ведёт на редирект: уровень доверия считается по домену издания."""
    docs = news.parse_feed(raw_fixture("news_bing_quantum_sensing.xml"), "quantum sensing", unwrap="bing")

    assert docs
    for doc in docs:
        assert "bing.com" not in doc.url
        assert doc.source and "bing" not in doc.source
        assert doc.id.startswith(f"{doc.source}:")


def test_unwrapped_link_without_target_is_dropped():
    """Развернуть не удалось — документ не берём: агрегатор сам по себе основанием быть не может."""
    feed = """<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>Заголовок</title><link>https://www.bing.com/news/apiclick.aspx?ref=FexRss</link></item>
    </channel></rss>"""

    assert news.parse_feed(feed, "quantum sensing", unwrap="bing") == []


def test_item_without_link_is_skipped():
    feed = """<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>Без ссылки</title></item>
    <item><title>Со ссылкой</title><link>https://example.org/a</link></item>
    </channel></rss>"""

    docs = news.parse_feed(feed, "quantum sensing")

    assert [d.title for d in docs] == ["Со ссылкой"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mon, 14 Sep 2026 20:15:00 GMT", date(2026, 9, 14)),  # формат Bing
        ("Wed, 08 Jul 2026 20:29:50 +0000", date(2026, 7, 8)),  # формат изданий на WordPress
        ("вчера", None),  # дату не разобрали — документ берём без даты
    ],
)
def test_pubdate_formats(raw: str, expected: date | None):
    feed = f"""<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>Новость</title><link>https://example.org/a</link><pubDate>{raw}</pubDate></item>
    </channel></rss>"""

    assert news.parse_feed(feed, "фраза")[0].published == expected


async def test_failing_feed_does_not_break_the_rest(settings, monkeypatch: pytest.MonkeyPatch):
    """Одно издание отвечает 429 — остальные новости всё равно собираются, ошибка попадает в errors."""
    monkeypatch.setattr(
        news,
        "feeds",
        lambda: [
            {"name": "работает", "url": "https://example.org/?s={query}&feed=rss2"},
            {"name": "падает", "url": "https://example.com/?s={query}&feed=rss2"},
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "example.com" in str(request.url):
            return httpx.Response(429)
        return httpx.Response(200, content=raw_fixture("news_techcrunch_quantum_sensing.xml").encode("utf-8"))

    errors: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        docs = await news.search("quantum sensing", settings, client, errors)

    assert docs
    assert errors == ["news:падает"]

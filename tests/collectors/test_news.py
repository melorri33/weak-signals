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
    """Одно издание отвечает ошибкой — остальные новости всё равно собираются, а ошибка попадает в errors."""
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
            return httpx.Response(500)
        return httpx.Response(200, content=raw_fixture("news_techcrunch_quantum_sensing.xml").encode("utf-8"))

    errors: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        docs = await news.search("quantum sensing", settings, client, errors)

    assert docs
    assert errors == ["news:падает"]


async def test_russian_phrases_do_not_touch_english_feeds(settings, monkeypatch: pytest.MonkeyPatch):
    """Русская фраза в англоязычной ленте ничего не находит, а лимит запросов тратит."""
    monkeypatch.setattr(news, "feeds", lambda: [{"name": "издание", "url": "https://example.org/?s={query}"}])

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("к ленте не должно быть ни одного запроса")

    errors: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await news.search("модели на краю сети", settings, client, errors) == []
    assert errors == []


async def test_feed_answering_429_is_dropped_for_the_rest_of_the_run(settings, monkeypatch: pytest.MonkeyPatch):
    """Конвейер приходит с 15–20 фразами: после отказа ленту больше не трогаем."""
    monkeypatch.setattr(news, "feeds", lambda: [{"name": "устала", "url": "https://example.org/?s={query}"}])
    monkeypatch.setattr(news, "FEED_PAUSE_S", 0)
    news.forget_exhausted_feeds()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(429)

    errors: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        for phrase in ("edge ai", "on-device inference", "tiny models"):
            assert await news.search(phrase, settings, client, errors) == []

    assert len(calls) == 1  # спросили один раз, дальше не ходим
    assert errors == []  # это не ошибка источника, а его просьба не частить
    news.forget_exhausted_feeds()


def _rss(*links: str) -> str:
    """Лента из заданных ссылок — чтобы у каждой страницы были свои документы."""
    items = "".join(
        f"<item><title>Новость {n}</title><link>{link}</link><description>о {link}</description></item>"
        for n, link in enumerate(links)
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel>{items}</channel></rss>'


def _paged_handler(pages: dict[str, str], calls: list[str]):
    """Отвечает страницей по параметру first; без параметра — первая страница."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        first = request.url.params.get("first", "1")
        body = pages.get(first)
        if body == "429":
            return httpx.Response(429)
        return httpx.Response(200, content=(body or _rss()).encode("utf-8"))

    return handler


async def test_feed_pages_are_merged(settings, monkeypatch: pytest.MonkeyPatch):
    """Лента со страницами листается: документы всех страниц идут в выдачу.

    Первая страница новостного поиска — около девяти записей с семи сайтов, а источники
    датасета организаторов разбросаны по двумстам сайтам. Замер 22.09: страницы 2-5
    приносят сотни новых документов с полутора-двух сотен сайтов.
    """
    monkeypatch.setattr(news, "feeds", lambda: [
        {"name": "поиск", "url": "https://example.org/?q={query}", "pages": [1, 11, 21]},
    ])
    monkeypatch.setattr(news, "FEED_PAUSE_S", 0)
    news.forget_exhausted_feeds()
    calls: list[str] = []
    pages = {
        "1": _rss("https://a.example.com/1", "https://b.example.com/2"),
        "11": _rss("https://c.example.com/3"),
        "21": _rss("https://d.example.com/4"),
    }

    errors: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_paged_handler(pages, calls))) as client:
        docs = await news.search("edge ai", settings, client, errors)

    assert len(calls) == 3
    assert sorted(d.source for d in docs) == ["a.example.com", "b.example.com", "c.example.com", "d.example.com"]
    news.forget_exhausted_feeds()


async def test_paging_stops_when_the_feed_repeats_itself(settings, monkeypatch: pytest.MonkeyPatch):
    """Страница без новых документов — дальше листать некуда, запросы не тратим."""
    monkeypatch.setattr(news, "feeds", lambda: [
        {"name": "поиск", "url": "https://example.org/?q={query}", "pages": [1, 11, 21]},
    ])
    monkeypatch.setattr(news, "FEED_PAUSE_S", 0)
    news.forget_exhausted_feeds()
    calls: list[str] = []
    same = _rss("https://a.example.com/1")
    pages = {"1": same, "11": same, "21": _rss("https://z.example.com/9")}

    errors: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_paged_handler(pages, calls))) as client:
        docs = await news.search("edge ai", settings, client, errors)

    assert len(calls) == 2  # третью страницу не просили
    assert [d.source for d in docs] == ["a.example.com"]
    news.forget_exhausted_feeds()


async def test_refusal_mid_paging_keeps_what_was_already_received(settings, monkeypatch: pytest.MonkeyPatch):
    """429 на второй странице не отменяет первую.

    Отмена посреди шага, уносящая уже полученное, в этом проекте встречалась трижды.
    """
    monkeypatch.setattr(news, "feeds", lambda: [
        {"name": "поиск", "url": "https://example.org/?q={query}", "pages": [1, 11, 21]},
    ])
    monkeypatch.setattr(news, "FEED_PAUSE_S", 0)
    news.forget_exhausted_feeds()
    calls: list[str] = []
    pages = {"1": _rss("https://a.example.com/1"), "11": "429", "21": _rss("https://z.example.com/9")}

    errors: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_paged_handler(pages, calls))) as client:
        docs = await news.search("edge ai", settings, client, errors)

    assert [d.source for d in docs] == ["a.example.com"]
    assert len(calls) == 2  # после отказа дальше не листаем
    assert errors == []
    news.forget_exhausted_feeds()

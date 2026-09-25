"""Транспорт GigaChat: токен, схема в системном сообщении, ошибки — офлайн на подменённой сети."""

import json
import time

import httpx
import pytest

from src.llm.errors import LLMError
from src.llm.providers import GigaChatBackend, make_backend
from src.llm.providers import gigachat as giga_module

_REAL_ASYNC_CLIENT = httpx.AsyncClient
KEY = "dGVzdDp0ZXN0"  # base64 «test:test», как настоящий ключ авторизации


@pytest.fixture(autouse=True)
def giga_env(monkeypatch: pytest.MonkeyPatch):
    """Ключ как из .env, но без .env; токены прошлых тестов забыты."""
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", KEY)
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE", "")
    monkeypatch.setattr(giga_module, "_tokens", {})


def _network(monkeypatch: pytest.MonkeyPatch, content: str = '{"city": "Томск"}', expires_in_s: float = 1800):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/oauth"):
            expires = int((time.time() + expires_in_s) * 1000)
            return httpx.Response(200, json={"access_token": f"tok{len(seen)}", "expires_at": expires})
        body = {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body)

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return seen


def test_only_models_from_the_tz_list():
    assert isinstance(make_backend("gigachat", "GigaChat-2"), GigaChatBackend)
    with pytest.raises(LLMError, match="не из списка ТЗ"):
        make_backend("gigachat", "GigaChat-Max")


def test_without_key_says_what_to_fill(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "")
    with pytest.raises(LLMError, match="GIGACHAT_CREDENTIALS"):
        make_backend("gigachat", "GigaChat-2")


async def test_token_is_reused_and_schema_goes_to_system(monkeypatch: pytest.MonkeyPatch):
    seen = _network(monkeypatch, content='```json\n{"city": "Томск"}\n```')
    backend = make_backend("gigachat", "GigaChat-2")
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    messages = [{"role": "system", "content": "Ты помощник."}, {"role": "user", "content": "Город?"}]

    first = await backend.chat(messages, schema, max_tokens=50, timeout_s=5)
    await backend.chat(messages, schema, max_tokens=50, timeout_s=5)

    assert first == '{"city": "Томск"}'  # обёртка ``` снята
    assert [r.url.path.endswith("/oauth") for r in seen] == [True, False, False]
    assert seen[0].headers["Authorization"] == f"Basic {KEY}" and seen[0].headers["RqUID"]
    body = json.loads(seen[1].content)
    assert body["model"] == "GigaChat-2" and "response_format" not in body
    assert body["messages"][0]["content"].startswith("Ты помощник.") and '"city"' in body["messages"][0]["content"]
    assert seen[1].headers["Authorization"] == "Bearer tok1"


async def test_expiring_token_is_renewed(monkeypatch: pytest.MonkeyPatch):
    seen = _network(monkeypatch, expires_in_s=10)  # меньше запаса TOKEN_MARGIN_S
    backend = make_backend("gigachat", "GigaChat-2")
    await backend.chat([{"role": "user", "content": "1"}], None, max_tokens=5, timeout_s=5)
    await backend.chat([{"role": "user", "content": "1"}], None, max_tokens=5, timeout_s=5)
    assert sum(r.url.path.endswith("/oauth") for r in seen) == 2

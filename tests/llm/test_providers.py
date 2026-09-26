"""Выбор провайдера и облачный транспорт: тела запросов, ошибки и журнал моделей — всё офлайн."""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from src.common.config import Settings
from src.common.logs import collected_model_calls, start_run_log
from src.llm.client import LLMClient
from src.llm.errors import LLMError
from src.llm.providers import OllamaBackend, YandexGPTBackend, make_backend
from src.llm.providers import ollama as ollama_module
from src.llm.providers import yandexgpt as yandex_module

# Настоящий класс клиента: подмена делается несколько раз за тест, и брать его из httpx уже нельзя.
_REAL_ASYNC_CLIENT = httpx.AsyncClient

MODEL = "yandexgpt-5-lite"
KEY = "test-api-key-123"  # заголовки HTTP только ascii, настоящий ключ тоже
FOLDER = "b1gtestfolder42"


class Answer(BaseModel):
    city: str


@pytest.fixture
def cloud_settings(monkeypatch: pytest.MonkeyPatch):
    """Ключи облака как из .env, но без .env: настройки общие, из src/common/config.py."""
    settings = Settings(yandex_api_key=KEY, yandex_folder_id=FOLDER)
    monkeypatch.setattr(yandex_module, "get_settings", lambda: settings)
    return settings


def _mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    """Подменить сеть заглушкой и вернуть список запросов, которые до неё дошли."""
    seen: list[httpx.Request] = []

    def factory(*args, **kwargs):
        def wrapped(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        kwargs["transport"] = httpx.MockTransport(wrapped)
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return seen


def _cloud_answer(content: str, finish_reason: str = "stop") -> httpx.Response:
    body = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}]
    }
    return httpx.Response(200, json=body)


def test_ollama_is_the_default_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ollama_module, "get_settings", lambda: Settings(ollama_url="http://localhost:11434"))
    monkeypatch.setattr(ollama_module, "DOCKER_MARKER", tmp_path / "не-контейнер")
    backend = make_backend("ollama", "qwen3:8b")
    assert isinstance(backend, OllamaBackend)
    assert backend.provider == "ollama" and backend.model == "qwen3:8b"
    assert backend.base_url == "http://localhost:11434"


def test_in_container_ollama_is_looked_up_at_another_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Внутри контейнера localhost — это сам контейнер, и модель на хосте так не найти."""
    marker = tmp_path / ".dockerenv"
    marker.touch()
    monkeypatch.setattr(ollama_module, "DOCKER_MARKER", marker)
    monkeypatch.setattr(
        ollama_module,
        "get_settings",
        lambda: Settings(ollama_url="http://localhost:11434", ollama_url_in_docker="http://host.docker.internal:11434"),
    )
    assert make_backend("ollama", "qwen3:8b").base_url == "http://host.docker.internal:11434"


def test_unknown_provider_names_what_is_available():
    with pytest.raises(LLMError) as exc:
        make_backend("openrouter", "gpt-4.1")
    assert "ollama" in str(exc.value) and "yandexgpt" in str(exc.value)


def test_cloud_model_outside_the_list_is_refused(cloud_settings):
    """ТЗ разрешает только свой перечень облачных моделей — проверяем до вызова сети."""
    with pytest.raises(LLMError) as exc:
        make_backend("yandexgpt", "gpt-4o-mini")
    assert "не из списка ТЗ" in str(exc.value)


def test_cloud_without_keys_says_which_variables_to_fill(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(yandex_module, "get_settings", lambda: Settings(yandex_api_key="", yandex_folder_id=""))
    with pytest.raises(LLMError) as exc:
        make_backend("yandexgpt", MODEL)
    assert "YANDEX_API_KEY" in str(exc.value) and "YANDEX_FOLDER_ID" in str(exc.value)


def test_yandex_alias_works(cloud_settings):
    assert isinstance(make_backend("Yandex", MODEL), YandexGPTBackend)


async def test_cloud_request_carries_key_folder_and_schema(cloud_settings, monkeypatch: pytest.MonkeyPatch):
    seen = _mock_transport(monkeypatch, lambda request: _cloud_answer('{"city": "Афины"}'))
    backend = make_backend("yandexgpt", MODEL)
    content = await backend.chat(
        messages=[{"role": "user", "content": "Где были первые Олимпийские игры?"}],
        json_schema=Answer.model_json_schema(),
        max_tokens=50,
        timeout_s=10,
    )
    assert content == '{"city": "Афины"}'

    request = seen[0]
    assert request.headers["authorization"] == f"Api-Key {KEY}"
    assert request.headers["openai-project"] == FOLDER
    body = json.loads(request.content)
    assert body["model"] == f"gpt://{FOLDER}/{MODEL}"
    assert body["max_tokens"] == 50 and body["stream"] is False
    # Схема — текстом в системном сообщении: structured output Яндекса ломает ключи на вложенных списках.
    assert "response_format" not in body
    assert '"city"' in body["messages"][0]["content"]


async def test_cloud_error_shows_status_and_answer_text(cloud_settings, monkeypatch: pytest.MonkeyPatch):
    """По одному коду не понять, дело в ключе, каталоге или схеме, — текст ответа нужен в ошибке."""
    _mock_transport(monkeypatch, lambda request: httpx.Response(401, text="unauthorized: bad api key"))
    backend = make_backend("yandexgpt", MODEL)
    with pytest.raises(LLMError) as exc:
        await backend.chat(messages=[{"role": "user", "content": "тест"}], json_schema=None, max_tokens=5, timeout_s=5)
    assert "401" in str(exc.value) and "bad api key" in str(exc.value)


async def test_cloud_health_check_is_one_short_request(cloud_settings, monkeypatch: pytest.MonkeyPatch):
    seen = _mock_transport(monkeypatch, lambda request: _cloud_answer("1"))
    assert await make_backend("yandexgpt", MODEL).is_available() is True
    assert json.loads(seen[0].content)["max_tokens"] == 1

    _mock_transport(monkeypatch, lambda request: httpx.Response(500, text="internal"))
    assert await make_backend("yandexgpt", MODEL).is_available() is False


async def test_journal_records_cloud_model_and_provider(cloud_settings, monkeypatch: pytest.MonkeyPatch):
    """Требование ТЗ: в журнале видно, какая модель и какой провайдер отработали на шаге."""
    _mock_transport(monkeypatch, lambda request: _cloud_answer('{"city": "Афины"}'))
    monkeypatch.setattr(
        "src.llm.client.get_settings",
        lambda: Settings(llm_provider="yandexgpt", llm_model=MODEL, llm_cache=False),
    )
    start_run_log()
    answer = await LLMClient.from_settings().ask_json(step="expand_query", prompt="тест", schema=Answer)
    assert answer.city == "Афины"

    calls = collected_model_calls()
    assert [(c.step, c.model, c.provider) for c in calls] == [("expand_query", MODEL, "yandexgpt")]

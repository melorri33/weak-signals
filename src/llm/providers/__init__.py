"""Транспорты к моделям: локальный Ollama и облако из списка ТЗ (YandexGPT, GigaChat).

Выбор провайдера — переменная `LLM_PROVIDER`. Клиент (src/llm/client.py) один для всех: он хранит
кэш ответов, повтор при непрохождении схемы и запись в журнал моделей, а транспорт отвечает только
за один сетевой вызов. Так добавление ещё одного провайдера не трогает конвейер.

Требование ТЗ: облачные модели — только из списка организаторов. Список зашит в каждом транспорте,
и модель не из списка — это ошибка на старте, а не тихий вызов чего попало.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.llm.errors import LLMError
from src.llm.providers.gigachat import GigaChatBackend
from src.llm.providers.ollama import OllamaBackend
from src.llm.providers.yandexgpt import YandexGPTBackend


@runtime_checkable
class ChatBackend(Protocol):
    """Один сетевой вызов чата. Всё остальное — забота клиента."""

    model: str
    provider: str

    async def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict | None,
        max_tokens: int,
        timeout_s: float,
    ) -> str:
        """Ответ модели текстом. Не ответила — LLMError."""
        ...

    async def is_available(self) -> bool:
        """Готова ли модель к работе (для GET /health)."""
        ...


# Имя провайдера в .env → транспорт. Псевдонимы — чтобы «yandex» тоже работало.
BACKENDS = {
    "ollama": OllamaBackend,
    "yandexgpt": YandexGPTBackend,
    "yandex": YandexGPTBackend,
    "gigachat": GigaChatBackend,
}


def make_backend(provider: str, model: str) -> ChatBackend:
    """Транспорт по имени провайдера из настроек."""
    factory = BACKENDS.get(provider.strip().lower())
    if factory is None:
        raise LLMError(
            f"Провайдер '{provider}' не поддерживается. Доступны: {', '.join(sorted(BACKENDS))}. "
            "Облачные модели — только из списка ТЗ."
        )
    return factory.from_settings(model)


__all__ = ["BACKENDS", "ChatBackend", "GigaChatBackend", "OllamaBackend", "YandexGPTBackend", "make_backend"]

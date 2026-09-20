"""Единый клиент LLM. Все вызовы моделей в проекте идут через него.

Клиент не знает, где живёт модель: сетевой вызов делает транспорт из src/llm/providers, который
выбирается по LLM_PROVIDER (`ollama` — локально, `yandexgpt` — облако из списка ТЗ). Здесь остаётся
то, что одинаково для любой модели: кэш ответов, повторная попытка при непрохождении схемы
и запись вызова в журнал моделей (требование ТЗ).

Использование:
    client = LLMClient.from_settings()
    phrases = await client.ask_json(step="expand_query", prompt=text, schema=Phrases)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.common.config import get_settings
from src.common.logs import get_logger, model_timer
from src.llm.errors import LLMError
from src.llm.providers import ChatBackend, make_backend

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Таймауты и параметры генерации держим здесь: в Settings их добавит владелец ядра, если понадобится.
REQUEST_TIMEOUT_S = 180.0
# Потолок длины ответа. Замер показал: без JSON-схемы модель не останавливается и упирается
# в потолок — при 2048 токенах это 4 минуты на один вызов. Держим потолок близко к нужной длине
# и поднимаем его точечно там, где ответ действительно длинный (подробный отчёт по сигналу).
NUM_PREDICT = 800

# Сколько раз просим модель переделать ответ, если он не прошёл проверку схемой.
RETRIES = 1


@dataclass(frozen=True)
class LLMClient:
    """Клиент чата: схема, кэш и журнал моделей. Куда идёт запрос — решает транспорт."""

    model: str
    provider: str
    backend: ChatBackend

    @classmethod
    def from_settings(cls) -> LLMClient:
        s = get_settings()
        backend = make_backend(s.llm_provider, s.llm_model)
        return cls(model=backend.model, provider=backend.provider, backend=backend)

    async def ask_json(
        self,
        step: str,
        prompt: str,
        schema: type[T],
        system: str | None = None,
        max_tokens: int = NUM_PREDICT,
    ) -> T:
        """Спросить модель и получить ответ, разобранный в модель pydantic.

        Схему передаём и в Ollama, и в облако, но гарантий ни один провайдер не даёт — поэтому
        проверяем ответ сами и при неудаче делаем одну повторную попытку с указанием на ошибку.
        """
        messages = _messages(prompt, system)
        json_schema = schema.model_json_schema()
        last_error = ""
        for attempt in range(RETRIES + 1):
            raw = await self._chat(step=step, messages=messages, json_schema=json_schema, max_tokens=max_tokens)
            try:
                return schema.model_validate_json(raw)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                log.warning("step=%s попытка %d: ответ не прошёл схему: %s", step, attempt + 1, last_error[:300])
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Ответ не прошёл проверку схемой. Ошибка:\n"
                            f"{last_error[:500]}\n"
                            "Пришли только исправленный JSON по схеме, без пояснений."
                        ),
                    },
                ]
        raise LLMError(f"шаг {step}: ответ не прошёл схему после {RETRIES + 1} попыток: {last_error[:300]}")

    async def ask_text(self, step: str, prompt: str, system: str | None = None, max_tokens: int = NUM_PREDICT) -> str:
        """Свободный текст без схемы (например, подробный отчёт по сигналу в markdown)."""
        return await self._chat(step=step, messages=_messages(prompt, system), json_schema=None, max_tokens=max_tokens)

    async def is_available(self) -> bool:
        """Готова ли модель к работе (для GET /health)."""
        return await self.backend.is_available()

    async def _chat(
        self,
        step: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None,
        max_tokens: int = NUM_PREDICT,
    ) -> str:
        """Один вызов модели. Пишет запись в журнал моделей даже если вызов упал.

        Одинаковый запрос к одной модели даёт одинаковый ответ (температура 0.2), поэтому ответ
        кэшируется на диске: повторный прогон и демонстрация не ждут модель по новой.
        """
        cache_key = _cache_key(self.model, messages, json_schema, max_tokens)
        if (cached := _cache_get(cache_key)) is not None:
            log.info("step=%s ответ взят из кэша", step)
            return cached
        try:
            with model_timer(step=step, model=self.model, provider=self.provider):
                content = await self.backend.chat(
                    messages=messages,
                    json_schema=json_schema,
                    max_tokens=max_tokens,
                    timeout_s=REQUEST_TIMEOUT_S,
                )
        except LLMError as exc:
            raise LLMError(f"шаг {step}: {exc}") from exc
        if not content.strip():
            raise LLMError(f"шаг {step}: модель вернула пустой ответ")
        _cache_put(cache_key, content)
        return content


# Кэш ответов модели. Лежит в data/ (папка не коммитится). Ключ — модель + запрос целиком,
# поэтому правка промпта автоматически делает старые ответы недействительными.
CACHE_DIR = Path("data") / "llm_cache"


def _cache_key(model: str, messages: list[dict[str, str]], json_schema: dict[str, Any] | None, max_tokens: int) -> str:
    raw = json.dumps([model, messages, json_schema, max_tokens], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    if not get_settings().llm_cache:
        return None
    path = CACHE_DIR / f"{key}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _cache_put(key: str, content: str) -> None:
    if not get_settings().llm_cache:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{key}.txt").write_text(content, encoding="utf-8")
    except OSError as exc:  # кэш — удобство, а не обязательное условие работы
        log.warning("Не смог записать кэш ответа модели: %s", exc)


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}] if system else []
    return [*messages, {"role": "user", "content": prompt}]


def dumps_ru(data: Any) -> str:
    """JSON для вставки в промпт: русский текст читаемым, без escape-последовательностей."""
    return json.dumps(data, ensure_ascii=False, indent=None)

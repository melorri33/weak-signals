"""Единый клиент локальной LLM (Ollama). Все вызовы моделей в проекте идут через него.

Почему один класс: ТЗ разрешает облачные модели только из своего списка и только после
согласования. Когда/если понадобится облако — меняем реализацию здесь, остальной код не трогаем.

Использование:
    client = LLMClient.from_settings()
    phrases = await client.ask_json(step="expand_query", prompt=text, schema=Phrases)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.common.config import get_settings
from src.common.logs import get_logger, model_timer

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Таймауты и параметры генерации держим здесь: в Settings их добавит владелец ядра, если понадобится.
REQUEST_TIMEOUT_S = 180.0
PING_TIMEOUT_S = 3.0
TEMPERATURE = 0.2
NUM_PREDICT = 2048
KEEP_ALIVE = "10m"

# Сколько раз просим модель переделать ответ, если он не прошёл проверку схемой.
RETRIES = 1


class LLMError(RuntimeError):
    """LLM не ответила или ответ не прошёл проверку схемой после повторной попытки."""


@dataclass(frozen=True)
class LLMClient:
    """Клиент чата с локальной моделью через Ollama."""

    model: str
    provider: str
    base_url: str

    @classmethod
    def from_settings(cls) -> LLMClient:
        s = get_settings()
        if s.llm_provider != "ollama":
            raise LLMError(
                f"Провайдер '{s.llm_provider}' не поддерживается: пока работаем только на локальном Ollama. "
                "Облачные модели — только из списка ТЗ и после согласования с организаторами."
            )
        return cls(model=s.llm_model, provider=s.llm_provider, base_url=s.ollama_url.rstrip("/"))

    async def ask_json(self, step: str, prompt: str, schema: type[T], system: str | None = None) -> T:
        """Спросить модель и получить ответ, разобранный в модель pydantic.

        Ollama принимает JSON-схему в поле format, но гарантий нет — поэтому проверяем сами
        и при неудаче делаем одну повторную попытку с указанием на ошибку.
        """
        messages = _messages(prompt, system)
        json_schema = schema.model_json_schema()
        last_error = ""
        for attempt in range(RETRIES + 1):
            raw = await self._chat(step=step, messages=messages, json_schema=json_schema)
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

    async def ask_text(self, step: str, prompt: str, system: str | None = None) -> str:
        """Свободный текст без схемы (например, подробный отчёт по сигналу в markdown)."""
        return await self._chat(step=step, messages=_messages(prompt, system), json_schema=None)

    async def is_available(self) -> bool:
        """Живёт ли Ollama и загружена ли наша модель (для GET /health)."""
        try:
            async with self._http(PING_TIMEOUT_S) as http:
                response = await http.get("/api/tags")
                response.raise_for_status()
                models = [m.get("model", "") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("Ollama недоступна на %s: %s", self.base_url, exc)
            return False
        if self.model not in models:
            log.warning("Модель %s не скачана. Запусти: ollama pull %s", self.model, self.model)
            return False
        return True

    async def _chat(self, step: str, messages: list[dict[str, str]], json_schema: dict[str, Any] | None) -> str:
        """Один вызов /api/chat. Пишет запись в журнал моделей даже если вызов упал."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            # Qwen3 по умолчанию «рассуждает» — это втрое дольше. Для наших задач это не нужно.
            "think": False,
            "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
        }
        if json_schema is not None:
            body["format"] = json_schema
        try:
            with model_timer(step=step, model=self.model, provider=self.provider):
                data = await self._post_chat(body)
        except httpx.HTTPError as exc:
            raise LLMError(f"шаг {step}: Ollama не ответила ({exc})") from exc
        content = (data.get("message") or {}).get("content", "")
        if not content.strip():
            raise LLMError(f"шаг {step}: модель вернула пустой ответ")
        return content

    async def _post_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /api/chat. Старые сборки Ollama не знают про think — тогда повторяем без него."""
        async with self._http(REQUEST_TIMEOUT_S) as http:
            response = await http.post("/api/chat", json=body)
            if response.status_code == httpx.codes.BAD_REQUEST and "think" in response.text.lower():
                log.info("Эта сборка Ollama не принимает think — повторяю запрос без него")
                response = await http.post("/api/chat", json={k: v for k, v in body.items() if k != "think"})
            response.raise_for_status()
            return response.json()

    def _http(self, timeout_s: float) -> httpx.AsyncClient:
        # trust_env=False: если на машине настроен системный прокси, локальный Ollama через него не ходит.
        return httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s, trust_env=False)


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}] if system else []
    return [*messages, {"role": "user", "content": prompt}]


def dumps_ru(data: Any) -> str:
    """JSON для вставки в промпт: русский текст читаемым, без escape-последовательностей."""
    return json.dumps(data, ensure_ascii=False, indent=None)

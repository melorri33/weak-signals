"""Локальная модель через Ollama: POST /api/chat.

Основной вариант проекта — модель работает на своей машине, ничего не уходит наружу и не стоит денег.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.common.config import get_settings
from src.common.logs import get_logger
from src.llm.errors import LLMError

log = get_logger(__name__)

# Признак контейнера: файл создаёт сам Docker. Внутри контейнера localhost — это он сам,
# поэтому адрес Ollama берём из отдельной настройки (OLLAMA_URL_IN_DOCKER), иначе конвейер
# в контейнере ищет модель у себя и не находит, хотя на хосте она работает.
DOCKER_MARKER = Path("/.dockerenv")

PING_TIMEOUT_S = 3.0
KEEP_ALIVE = "10m"
TEMPERATURE = 0.2


@dataclass(frozen=True)
class OllamaBackend:
    """Чат с локальной моделью. `model` — как в `ollama list`, например qwen3:8b."""

    model: str
    base_url: str
    provider: str = "ollama"

    @classmethod
    def from_settings(cls, model: str) -> OllamaBackend:
        settings = get_settings()
        if DOCKER_MARKER.exists():
            log.info("Работаем в контейнере — Ollama ищем на %s", settings.ollama_url_in_docker)
            return cls(model=model, base_url=settings.ollama_url_in_docker.rstrip("/"))
        return cls(model=model, base_url=settings.ollama_url.rstrip("/"))

    async def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict | None,
        max_tokens: int,
        timeout_s: float,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            # Qwen3 по умолчанию «рассуждает» — это втрое дольше. Для наших задач это не нужно.
            "think": False,
            "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
        }
        if json_schema is not None:
            body["format"] = json_schema
        try:
            data = await self._post_chat(body, timeout_s)
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama не ответила ({exc})") from exc
        return (data.get("message") or {}).get("content", "")

    async def is_available(self) -> bool:
        """Живёт ли Ollama и скачана ли наша модель."""
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

    async def _post_chat(self, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """POST /api/chat. Старые сборки Ollama не знают про think — тогда повторяем без него."""
        async with self._http(timeout_s) as http:
            response = await http.post("/api/chat", json=body)
            if response.status_code == httpx.codes.BAD_REQUEST and "think" in response.text.lower():
                log.info("Эта сборка Ollama не принимает think — повторяю запрос без него")
                response = await http.post("/api/chat", json={k: v for k, v in body.items() if k != "think"})
            response.raise_for_status()
            return response.json()

    def _http(self, timeout_s: float) -> httpx.AsyncClient:
        # trust_env=False: если на машине настроен системный прокси, локальный Ollama через него не ходит.
        return httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s, trust_env=False)

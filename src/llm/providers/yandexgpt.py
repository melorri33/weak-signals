"""Облачная модель YandexGPT — запасной вариант, когда локальной модели нет.

Нужен по двум причинам: на машине без видеокарты локальный прогон идёт десятки минут, а к финалу
стенд должен работать у жюри, где Ollama никто ставить не будет.

Работаем через OpenAI-совместимый эндпоинт Yandex AI Studio: тело запроса и ответа такие же, как
у OpenAI, а авторизация — по ключу API (`Authorization: Api-Key …`) с идентификатором каталога
в заголовке `OpenAI-Project`. Структурированный ответ — `response_format.json_schema`, поэтому
проверка схемой работает так же, как у Ollama.
Документация: https://aistudio.yandex.ru/docs/ru/ai-studio/operations/generation/completions-structured

Что нужно в .env:
    LLM_PROVIDER=yandexgpt
    LLM_MODEL=yandexgpt-5-lite
    YANDEX_API_KEY=…            # сервисный аккаунт с ролью ai.languageModels.user
    YANDEX_FOLDER_ID=…          # каталог в облаке
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.logs import get_logger
from src.llm.errors import LLMError

log = get_logger(__name__)

TEMPERATURE = 0.2
PING_TIMEOUT_S = 15.0
# Сколько символов ответа сервиса показываем в ошибке: по коду 400 без тела непонятно, что не так.
ERROR_TEXT_LIMIT = 400

# Разрешённые ТЗ облачные модели Яндекса → имя модели в запросе. Модель не из списка — ошибка на старте:
# ТЗ разрешает облако только из своего перечня, и молча вызвать что-то другое нельзя.
ALLOWED_MODELS = {
    "yandexgpt-5-lite": "YandexGPT Lite 5",
    "yandexgpt-5-pro": "YandexGPT Pro 5",
    "yandexgpt-5.1": "YandexGPT Pro 5.1",
    # Прежние адреса тех же моделей — их всё ещё отдаёт документация.
    "yandexgpt/rc": "YandexGPT Lite 5",
    "yandexgpt/latest": "YandexGPT Pro 5",
}


class YandexSettings(BaseSettings):
    """Доступы к облаку.

    Живут здесь, а не в src/common/config.py: тот файл ведёт владелец ядра. Как только он перенесёт
    эти три переменные в Settings, модуль будет читать их оттуда, а .env менять не придётся.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_api_url: str = "https://ai.api.cloud.yandex.net/v1/chat/completions"


@lru_cache
def get_yandex_settings() -> YandexSettings:
    return YandexSettings()


@dataclass(frozen=True)
class YandexGPTBackend:
    """Чат с YandexGPT. `model` — короткое имя из ALLOWED_MODELS, оно же попадает в журнал моделей."""

    model: str
    folder_id: str
    api_key: str
    api_url: str
    provider: str = "yandexgpt"

    @classmethod
    def from_settings(cls, model: str) -> YandexGPTBackend:
        s = get_yandex_settings()
        model = model.strip()
        if model not in ALLOWED_MODELS:
            raise LLMError(
                f"Модель '{model}' не из списка ТЗ. Разрешены: {', '.join(sorted(ALLOWED_MODELS))}. "
                "Другие облачные модели — только после согласования с организаторами."
            )
        if not s.yandex_api_key or not s.yandex_folder_id:
            raise LLMError(
                "Для LLM_PROVIDER=yandexgpt нужны YANDEX_API_KEY и YANDEX_FOLDER_ID в .env "
                "(ключ сервисного аккаунта с ролью ai.languageModels.user и каталог облака)."
            )
        return cls(
            model=model,
            folder_id=s.yandex_folder_id,
            api_key=s.yandex_api_key,
            api_url=s.yandex_api_url,
        )

    @property
    def model_uri(self) -> str:
        """Адрес модели в облаке: gpt://<каталог>/<модель>."""
        return f"gpt://{self.folder_id}/{self.model}"

    async def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict | None,
        max_tokens: int,
        timeout_s: float,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model_uri,
            "messages": messages,
            "stream": False,
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": json_schema},
            }
        data = await self._post(body, timeout_s)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.model}: в ответе нет ни одного варианта ({_short(data)})")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            log.warning("%s: ответ обрезан по max_tokens=%d — возможно, не пройдёт схему", self.model, max_tokens)
        return (choice.get("message") or {}).get("content", "")

    async def is_available(self) -> bool:
        """Отвечает ли облако на самый короткий запрос (для GET /health)."""
        try:
            await self._post(
                {
                    "model": self.model_uri,
                    "messages": [{"role": "user", "content": "1"}],
                    "max_tokens": 1,
                    "stream": False,
                },
                PING_TIMEOUT_S,
            )
        except LLMError as exc:
            log.warning("YandexGPT недоступна: %s", exc)
            return False
        return True

    async def _post(self, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "OpenAI-Project": self.folder_id,
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as http:
                response = await http.post(self.api_url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.model}: облако не ответило ({exc})") from exc
        if response.status_code != httpx.codes.OK:
            # Текст ответа в ошибку: по одному коду 400 не понять, дело в схеме, ключе или каталоге.
            raise LLMError(f"{self.model}: облако вернуло {response.status_code} — {_short(response.text)}")
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(f"{self.model}: ответ облака не разобрался как JSON ({_short(response.text)})") from exc


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    return text[:ERROR_TEXT_LIMIT]

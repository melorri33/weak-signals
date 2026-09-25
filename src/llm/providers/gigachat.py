"""Облачная модель GigaChat (Сбер) — вторая облачная модель из списка ТЗ, рядом с YandexGPT.

Нужна, чтобы сравнить модели на одних и тех же запросах и выбрать ту, что лучше находит слабые сигналы.

Как устроен доступ (проверено живыми запросами 25.09.2026):
- ключ авторизации из личного кабинета меняется на токен: POST ngw.devices.sberbank.ru:9443/api/v2/oauth,
  заголовки `Authorization: Basic <ключ>` и `RqUID: <uuid4>`, тело `scope=GIGACHAT_API_PERS`.
  В ответе `access_token` и `expires_at` в миллисекундах; токен живёт 30 минут;
- чат: POST gigachat.devices.sberbank.ru/api/v1/chat/completions, тело и ответ как у OpenAI;
- сертификаты серверов выданы НУЦ Минцифры, которого нет в системном списке доверенных. Корневой
  сертификат кладём файлом и указываем путь в GIGACHAT_CA_BUNDLE; проверку TLS не отключаем;
- `response_format` с JSON-схемой сервис принимает, но не соблюдает: отвечает нумерованным списком.
  Поэтому схему передаём текстом в системном сообщении, а проверку схемой и повтор делает LLMClient.

Что нужно в .env:
    LLM_PROVIDER=gigachat
    LLM_MODEL=GigaChat-2                  # GigaChat 2 Lite; GigaChat-2-Pro, GigaChat-2-Max
    GIGACHAT_CREDENTIALS=…                # «Authorization Key» из developers.sber.ru → Настройки API
    GIGACHAT_CA_BUNDLE=data/certs/russian_trusted_root_ca_pem.crt

Настройки читаются здесь, а не в src/common/config.py: общий конфиг — зона ядра, перенос туда предложен владельцу.
"""

from __future__ import annotations

import json
import re
import ssl
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.logs import get_logger
from src.llm.errors import LLMError

log = get_logger(__name__)

TEMPERATURE = 0.2
PING_TIMEOUT_S = 15.0
TOKEN_TIMEOUT_S = 20.0
# Токен обновляем заранее: запрос, начатый за секунду до истечения, не должен получить 401.
TOKEN_MARGIN_S = 60.0
ERROR_TEXT_LIMIT = 400

# Разрешённые ТЗ модели → название в ТЗ. Остальные модели сервиса (GigaChat-Max, -Plus и т.п.) — не из списка.
ALLOWED_MODELS = {
    "GigaChat-2": "GigaChat 2 Lite",
    "GigaChat-2-Pro": "GigaChat 2 Pro",
    "GigaChat-2-Max": "GigaChat 2 Max",
}


class GigaChatSettings(BaseSettings):
    """Доступ к GigaChat из .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_ca_bundle: str = ""
    gigachat_auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_api_url: str = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


# Токен на процесс: клиент создаётся заново на каждый шаг конвейера, а просить токен на каждый вызов незачем.
_tokens: dict[str, tuple[str, float]] = {}


@dataclass(frozen=True)
class GigaChatBackend:
    """Чат с GigaChat. `model` — имя из ALLOWED_MODELS, оно же попадает в журнал моделей."""

    model: str
    credentials: str
    scope: str
    ca_bundle: str
    auth_url: str
    api_url: str
    provider: str = "gigachat"

    @classmethod
    def from_settings(cls, model: str) -> GigaChatBackend:
        s = GigaChatSettings()
        model = model.strip()
        if model not in ALLOWED_MODELS:
            raise LLMError(
                f"Модель '{model}' не из списка ТЗ. Разрешены: {', '.join(sorted(ALLOWED_MODELS))}. "
                "Другие облачные модели — только после согласования с организаторами."
            )
        if not s.gigachat_credentials:
            raise LLMError(
                "Для LLM_PROVIDER=gigachat нужен GIGACHAT_CREDENTIALS в .env "
                "(«Authorization Key» из developers.sber.ru → проект GigaChat API → Настройки API)."
            )
        return cls(
            model=model,
            credentials=s.gigachat_credentials.strip(),
            scope=s.gigachat_scope,
            ca_bundle=s.gigachat_ca_bundle,
            auth_url=s.gigachat_auth_url,
            api_url=s.gigachat_api_url,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict | None,
        max_tokens: int,
        timeout_s: float,
    ) -> str:
        if json_schema is not None:
            messages = _with_schema(messages, json_schema)
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
        }
        data = await self._post(body, timeout_s)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.model}: в ответе нет ни одного варианта ({_short(data)})")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            log.warning("%s: ответ обрезан по max_tokens=%d — возможно, не пройдёт схему", self.model, max_tokens)
        content = (choice.get("message") or {}).get("content", "")
        return _strip_fence(content) if json_schema is not None else content

    async def is_available(self) -> bool:
        """Отвечает ли облако на самый короткий запрос (для GET /health)."""
        try:
            await self._post(
                {"model": self.model, "messages": [{"role": "user", "content": "1"}], "max_tokens": 1},
                PING_TIMEOUT_S,
            )
        except LLMError as exc:
            log.warning("GigaChat недоступна: %s", exc)
            return False
        return True

    def _verify(self) -> ssl.SSLContext | bool:
        """Проверка TLS с корневым сертификатом Минцифры, если он указан; иначе — системный список."""
        if not self.ca_bundle:
            return True
        try:
            return ssl.create_default_context(cafile=self.ca_bundle)
        except OSError as exc:
            raise LLMError(f"Не читается сертификат GIGACHAT_CA_BUNDLE={self.ca_bundle}: {exc}") from exc

    async def _token(self, http: httpx.AsyncClient) -> str:
        cached = _tokens.get(self.credentials)
        if cached and cached[1] - TOKEN_MARGIN_S > time.time():
            return cached[0]
        headers = {
            "Authorization": f"Basic {self.credentials}",
            "RqUID": str(uuid.uuid4()),
            "Accept": "application/json",
        }
        try:
            response = await http.post(
                self.auth_url, data={"scope": self.scope}, headers=headers, timeout=TOKEN_TIMEOUT_S
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.model}: не получили токен ({_tls_hint(exc)})") from exc
        if response.status_code != httpx.codes.OK:
            raise LLMError(f"{self.model}: токен не выдан, {response.status_code} — {_short(response.text)}")
        data = response.json()
        token, expires_ms = data.get("access_token"), data.get("expires_at")
        if not token or not expires_ms:
            raise LLMError(f"{self.model}: в ответе на запрос токена нет access_token или expires_at")
        _tokens[self.credentials] = (token, expires_ms / 1000)
        return token

    async def _post(self, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=timeout_s, verify=self._verify()) as http:
                headers = {"Authorization": f"Bearer {await self._token(http)}", "Accept": "application/json"}
                response = await http.post(self.api_url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.model}: облако не ответило ({_tls_hint(exc)})") from exc
        if response.status_code == httpx.codes.UNAUTHORIZED:
            _tokens.pop(self.credentials, None)  # токен отозван раньше срока — следующий вызов возьмёт новый
        if response.status_code != httpx.codes.OK:
            raise LLMError(f"{self.model}: облако вернуло {response.status_code} — {_short(response.text)}")
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(f"{self.model}: ответ облака не разобрался как JSON ({_short(response.text)})") from exc


def _with_schema(messages: list[dict[str, str]], json_schema: dict) -> list[dict[str, str]]:
    """Добавить JSON-схему в системное сообщение: сам сервис схему не соблюдает."""
    rule = "Ответ — только один JSON-объект строго по этой JSON-схеме, без пояснений и без ```:\n" + json.dumps(
        json_schema, ensure_ascii=False
    )
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": f"{messages[0]['content']}\n\n{rule}"}, *messages[1:]]
    return [{"role": "system", "content": rule}, *messages]


def _strip_fence(content: str) -> str:
    """Модель иногда оборачивает JSON в ```json … ``` — схема такое не пропустит."""
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, re.DOTALL)
    return match.group(1) if match else content


def _tls_hint(exc: Exception) -> str:
    text = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" in text:
        return f"{text}; нужен корневой сертификат Минцифры в GIGACHAT_CA_BUNDLE"
    return text


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    return text[:ERROR_TEXT_LIMIT]

"""Общее для всех источников: заголовки, вежливые паузы, безопасный вызов."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from src.common.config import Settings
from src.common.logs import get_logger

log = get_logger(__name__)

T = TypeVar("T")


_PROJECT_URL = "https://github.com/melorri33/weak-signals"


def user_agent(purpose: str, settings: Settings) -> str:
    """User-Agent с рабочим контактом — источники без него банят (Википедия отвечает 403 на «пустой» UA).

    Ссылка на репозиторий всегда настоящая и не зависит от того, заполнен ли CONTACT_EMAIL.
    """
    contact = f"; mailto:{settings.contact_email}" if settings.contact_email else ""
    return f"weak-signals/0.1 ({_PROJECT_URL}; {purpose}{contact})"


class RateLimiter:
    """Не чаще одного запроса в `interval_s` секунд для источника, который это требует (arXiv)."""

    def __init__(self, interval_s: float) -> None:
        self._interval_s = interval_s
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._interval_s:
                await asyncio.sleep(self._interval_s - elapsed)
            self._last_call = time.monotonic()


async def safe_call(source: str, call: Callable[[], Awaitable[T]], errors: list[str]) -> T | None:
    """Выполнить запрос к источнику; падение или таймаут — в лог и в `errors`, не наружу."""
    try:
        return await call()
    except Exception as exc:  # источник не должен валить весь сбор
        log.warning("источник %s не ответил: %s", source, exc)
        errors.append(source)
        return None

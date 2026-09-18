"""Логирование и журнал вызовов моделей (ТЗ требует явно логировать, какая модель что делала).

Использование:
    from src.common.logs import get_logger, log_model_call, model_timer

    with model_timer(step="expand_query", model="qwen3:8b", provider="ollama"):
        answer = client.chat(...)

    # в конвейере — собрать все вызовы текущего прогона:
    start_run_log()
    ...
    calls = collected_model_calls()
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from src.common.schemas import ModelCall

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

_run_calls: ContextVar[list[ModelCall] | None] = ContextVar("_run_calls", default=None)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


_models_log = get_logger("models")


def start_run_log() -> None:
    """Начать новый журнал вызовов моделей для текущего прогона."""
    _run_calls.set([])


def collected_model_calls() -> list[ModelCall]:
    """Все вызовы моделей текущего прогона (для SearchResult.model_calls)."""
    return list(_run_calls.get() or [])


def log_model_call(step: str, model: str, provider: str, duration_ms: int) -> ModelCall:
    """Записать один вызов модели в лог и в журнал текущего прогона."""
    call = ModelCall(step=step, model=model, provider=provider, duration_ms=duration_ms)
    _models_log.info("step=%s model=%s provider=%s duration_ms=%d", step, model, provider, duration_ms)
    calls = _run_calls.get()
    if calls is not None:
        calls.append(call)
    return call


@contextmanager
def model_timer(step: str, model: str, provider: str) -> Iterator[None]:
    """Замерить время блока и записать вызов модели (даже если блок упал)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log_model_call(step, model, provider, int((time.perf_counter() - t0) * 1000))

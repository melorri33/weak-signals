"""Ошибка LLM — отдельным модулем, чтобы транспорты (src/llm/providers) не зависели от клиента."""

from __future__ import annotations


class LLMError(RuntimeError):
    """Модель не ответила, отказала или её ответ не прошёл проверку схемой."""

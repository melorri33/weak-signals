"""Расширение запроса пользователя в поисковые фразы: `expand_query(query) -> list[str]`.

Работает через локальную LLM. Если Ollama недоступна или ответ негодный — отдаём простые
фразы, собранные из самого запроса, чтобы конвейер всё равно прошёл (требование: падение
одного шага не валит прогон).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.common.logs import get_logger
from src.llm.client import LLMClient, LLMError
from src.llm.prompt_loader import render

log = get_logger(__name__)

MIN_PHRASES = 8
MAX_PHRASES = 18
MAX_WORDS_IN_PHRASE = 6

# Слова, из-за которых поиск сползает в рекламу и обзоры вместо исследований.
_BANNED_WORDS = {"тренд", "тренды", "перспективный", "перспективные", "будущее", "прорыв", "trend", "trends", "future"}

# Чем дополняем запрос, если LLM недоступна. Не перевод, а грубая подстраховка.
_RU_SUFFIXES = ("", "ранние исследования", "пилотные проекты", "патенты", "методы", "прототипы", "архитектура")
_EN_SUFFIXES = ("emerging technology", "early stage research", "novel method", "preprint", "patent")


class _Phrases(BaseModel):
    """Схема ответа модели."""

    phrases: list[str] = Field(default_factory=list)


async def expand_query(query: str, client: LLMClient | None = None) -> list[str]:
    """Получить 10–18 поисковых фраз на русском и английском по свободному запросу."""
    query = query.strip()
    if not query:
        return []
    try:
        client = client or LLMClient.from_settings()
        answer = await client.ask_json(step="expand_query", prompt=render("expand_query", query=query), schema=_Phrases)
        phrases = _clean(answer.phrases)
    except LLMError as exc:
        log.warning("expand_query: LLM не помогла (%s) — беру фразы из запроса", exc)
        phrases = []
    if len(phrases) < MIN_PHRASES:
        log.warning("expand_query: годных фраз %d, добираю простыми вариантами запроса", len(phrases))
        phrases = _clean([*phrases, *_fallback_phrases(query)])
    return phrases[:MAX_PHRASES]


def _clean(phrases: list[str]) -> list[str]:
    """Убрать мусор, дубли и слишком длинные фразы, сохранив порядок."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in phrases:
        phrase = " ".join(raw.replace('"', " ").replace("«", " ").replace("»", " ").split())
        words = phrase.split()
        if not (1 < len(words) <= MAX_WORDS_IN_PHRASE):
            continue
        if any(w.strip(",.").lower() in _BANNED_WORDS for w in words):
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _fallback_phrases(query: str) -> list[str]:
    """Фразы без модели: запрос и запрос с уточнениями. Рекламные слова из запроса убираем."""
    base = " ".join(w for w in query.split() if w.strip(",.").lower() not in _BANNED_WORDS) or query
    ru = [f"{base} {suffix}".strip() for suffix in _RU_SUFFIXES]
    en = [f"{base} {suffix}".strip() for suffix in _EN_SUFFIXES]
    return [*ru, *en]

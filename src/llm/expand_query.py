"""Расширение запроса пользователя в поисковые фразы: `expand_query(query) -> list[str]`.

Работает через локальную LLM. Если Ollama недоступна или ответ негодный — отдаём простые
фразы, собранные из самого запроса, чтобы конвейер всё равно прошёл (требование: падение
одного шага не валит прогон).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.common.logs import get_logger
from src.llm.client import LLMClient, LLMError
from src.llm.prompt_loader import render

log = get_logger(__name__)

MIN_PHRASES = 8
MAX_PHRASES = 18
MAX_WORDS_IN_PHRASE = 6

# Сколько фраз латиницей нужно обязательно. Русская фраза бесполезна для половины конвейера:
# англоязычные ленты новостей её вообще не принимают (src/collectors/news.py пропускает кириллицу,
# иначе тратится лимит запросов впустую), а в arXiv и OpenAlex по ней почти ничего не находится.
# Прогон, где все фразы оказались русскими, собрал ноль документов и остановился с ошибкой.
MIN_LATIN_PHRASES = 6

_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)

# Слова, из-за которых поиск сползает в рекламу и обзоры вместо исследований.
_BANNED_WORDS = {"тренд", "тренды", "перспективный", "перспективные", "будущее", "прорыв", "trend", "trends", "future"}

# Служебные слова: на смысл фразы не влияют, но мешают заметить, что две фразы — про одно и то же
# («homomorphic encryption banking» и «homomorphic encryption for banking»).
_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "the",
    "to",
    "with",
    "using",
    "systems",
    "system",
    "applications",
    "application",
    "technologies",
    "technology",
    "в",
    "во",
    "и",
    "для",
    "из",
    "к",
    "на",
    "о",
    "об",
    "по",
    "при",
    "с",
    "со",
    "у",
    "системы",
    "система",
    "технологии",
    "технология",
    "применение",
}

# Модель иногда переносит в ответ подсказки из формата промпта («<русская фраза 1>») — их выкидываем.
# Ловим только угловые скобки и «фраза N» / «phrase N»: просто слово «phrase» встречается в нормальных
# терминах («key phrase extraction», «passphrase-less authentication»), и выкидывать их нельзя.
_PLACEHOLDER_RE = re.compile(r"[<>]|\b(?:фраза|phrase)\s*\d", re.IGNORECASE)

# Отрасли и общие направления: по такой фразе находятся обзоры рынка, а не конкретные технологии.
# Промпт их запрещает, но модель иногда всё равно копирует их из списка «не годится».
_TOO_BROAD = {
    "энергетика",
    "новые материалы",
    "ии в медицине",
    "роботы для склада",
    "edge computing",
    "edge ai",
    "искусственный интеллект",
    "машинное обучение",
    "artificial intelligence",
    "machine learning",
}

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
    if _count_latin(phrases) < MIN_LATIN_PHRASES:
        phrases = _clean([*phrases, *await _ask_for_english(query, phrases, client)])
    if len(phrases) < MIN_PHRASES:
        log.warning("expand_query: годных фраз %d, добираю простыми вариантами запроса", len(phrases))
        phrases = _clean([*phrases, *_fallback_phrases(query)])
    return phrases[:MAX_PHRASES]


def _is_latin(phrase: str) -> bool:
    """Фраза годится для англоязычных источников, если в ней нет кириллицы."""
    return _CYRILLIC_RE.search(phrase) is None


def _count_latin(phrases: list[str]) -> int:
    return sum(_is_latin(p) for p in phrases)


async def _ask_for_english(query: str, phrases: list[str], client: LLMClient | None) -> list[str]:
    """Добрать английские фразы отдельным вызовом.

    Промпт требует половину фраз на английском, но модель выполняет это через раз: на запросе про
    edge AI qwen3:14b в трёх прогонах из четырёх вернула только русские. Одного общего промпта мало,
    поэтому недостающие английские фразы просим отдельно — так проверяемо и видно в логе.
    Модель не ответила — берём грубую английскую подстраховку, лишь бы источники получили запрос.
    """
    need = MIN_LATIN_PHRASES - _count_latin(phrases)
    log.warning("expand_query: английских фраз %d из %d — добираю %d", _count_latin(phrases), MIN_LATIN_PHRASES, need)
    try:
        client = client or LLMClient.from_settings()
        answer = await client.ask_json(
            step="expand_query_en",
            prompt=render("expand_query_en", query=query, count=need + 2),
            schema=_Phrases,
        )
        english = [p for p in _clean(answer.phrases) if _is_latin(p)]
    except LLMError as exc:
        log.warning("expand_query: добор английских фраз не удался (%s)", exc)
        english = []
    if len(english) < need:
        english = [*english, *(p for p in _fallback_phrases(query) if _is_latin(p))]
    return english


def _clean(phrases: list[str]) -> list[str]:
    """Убрать мусор, дубли (в том числе фразы-близнецы) и слишком длинные фразы, сохранив порядок."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in phrases:
        phrase = " ".join(raw.replace('"', " ").replace("«", " ").replace("»", " ").split())
        words = phrase.split()
        if not (1 < len(words) <= MAX_WORDS_IN_PHRASE):
            continue
        if any(w.strip(",.").lower() in _BANNED_WORDS for w in words):
            continue
        if _looks_like_placeholder(phrase):
            log.warning("expand_query: модель вернула подсказку из формата промпта: %s", phrase)
            continue
        if phrase.lower() in _TOO_BROAD:
            log.warning("expand_query: фраза — целое направление, а не технология: %s", phrase)
            continue
        key = _dedup_key(phrase)
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _looks_like_placeholder(phrase: str) -> bool:
    """«<русская фраза 1>» — это не поисковая фраза, а скопированная подсказка из промпта."""
    return _PLACEHOLDER_RE.search(phrase) is not None


def _dedup_key(phrase: str) -> str:
    """Ключ склейки близнецов: без служебных слов и без порядка слов.

    «homomorphic encryption banking» и «homomorphic encryption for banking systems» дадут один ключ —
    искать по обеим бессмысленно, а место в наборе они занимают.
    """
    words = sorted(w.strip(",.-").lower() for w in phrase.split())
    meaningful = [w for w in words if w and w not in _FUNCTION_WORDS]
    return " ".join(meaningful or words)


def _fallback_phrases(query: str) -> list[str]:
    """Фразы без модели: запрос и запрос с уточнениями. Рекламные слова из запроса убираем."""
    base = " ".join(w for w in query.split() if w.strip(",.").lower() not in _BANNED_WORDS) or query
    ru = [f"{base} {suffix}".strip() for suffix in _RU_SUFFIXES]
    en = [f"{base} {suffix}".strip() for suffix in _EN_SUFFIXES]
    return [*ru, *en]

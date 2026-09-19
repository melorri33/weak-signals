"""Офлайн-тесты расширения запроса: чистка фраз и подстраховка без LLM."""

import pytest

from src.llm.client import LLMError
from src.llm.expand_query import MIN_PHRASES, _clean, expand_query


class _BrokenClient:
    """Клиент, как будто Ollama не запущена."""

    async def ask_json(self, **_: object) -> object:
        raise LLMError("Ollama недоступна")


class _GoodClient:
    """Клиент, который вернул фразы."""

    def __init__(self, phrases: list[str]) -> None:
        self.phrases = phrases

    async def ask_json(self, step: str, prompt: str, schema: type, system: str | None = None) -> object:
        return schema(phrases=self.phrases)


def test_clean_drops_junk():
    phrases = _clean(
        [
            "федеративное обучение в банках",
            "  федеративное   обучение в банках  ",  # дубль с лишними пробелами
            "«privacy-preserving credit scoring»",  # кавычки убираем
            "перспективные тренды финтеха",  # рекламное слово — выкидываем
            "скоринг",  # одно слово — не фраза
            "очень длинная фраза которая никуда не годится потому что слишком длинная",
        ]
    )
    assert phrases == ["федеративное обучение в банках", "privacy-preserving credit scoring"]


async def test_llm_phrases_used():
    client = _GoodClient(["федеративное обучение в банках", "privacy preserving credit scoring", "скоринг"])
    phrases = await expand_query("финтех", client=client)  # type: ignore[arg-type]

    # Фразы модели идут первыми, одиночное слово отброшено, остальное добрано из запроса.
    assert phrases[:2] == ["федеративное обучение в банках", "privacy preserving credit scoring"]
    assert len(phrases) >= MIN_PHRASES
    assert "финтех патенты" in phrases


async def test_fallback_without_llm():
    phrases = await expand_query("перспективные решения в финтехе", client=_BrokenClient())  # type: ignore[arg-type]

    assert len(phrases) >= MIN_PHRASES
    assert all(1 < len(p.split()) <= 6 for p in phrases)
    assert not any("перспективные" in p for p in phrases)  # рекламное слово из запроса убрано
    assert any(p.isascii() is False for p in phrases) and any("preprint" in p for p in phrases)  # RU + EN


@pytest.mark.parametrize("query", ["", "   "])
async def test_empty_query(query: str):
    assert await expand_query(query, client=_BrokenClient()) == []  # type: ignore[arg-type]

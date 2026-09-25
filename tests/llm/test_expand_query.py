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


def test_clean_drops_twin_phrases():
    """Фразы-близнецы отличаются только служебными словами — искать по обеим бессмысленно."""
    phrases = _clean(
        [
            "homomorphic encryption for banking systems",
            "homomorphic encryption banking",
            "banking homomorphic encryption applications",
            "homomorphic encryption for payments",  # другая подтема — остаётся
        ]
    )
    assert phrases == ["homomorphic encryption for banking systems", "homomorphic encryption for payments"]


def test_clean_drops_prompt_placeholders():
    """Модель иногда переносит в ответ подсказки из формата промпта."""
    assert _clean(["<русская фраза 1>", "<english phrase 2>", "натрий-ионные аккумуляторы"]) == [
        "натрий-ионные аккумуляторы"
    ]


def test_clean_drops_whole_directions():
    """Отрасль и общее направление — не поисковая фраза: по ним находятся обзоры рынка."""
    phrases = _clean(["edge computing", "машинное обучение", "микроконтроллеры с нейросетями"])
    assert phrases == ["микроконтроллеры с нейросетями"]


@pytest.mark.parametrize(
    "phrase",
    [
        "passphrase-less authentication",  # «phrase» внутри слова
        "key phrase extraction",  # «phrase» как отдельное слово, но без номера
        "фразовые эмбеддинги для поиска",  # русское слово с тем же корнем
    ],
)
def test_clean_keeps_terms_with_phrase_inside(phrase: str):
    """Слово «phrase» встречается в нормальных терминах — выкидывать их нельзя."""
    assert _clean([phrase]) == [phrase]


@pytest.mark.parametrize(
    "placeholder",
    [
        "<русская фраза 1>",
        "<english phrase 2>",
        "русская фраза 3",
        "english phrase 4",
        "фраза1",
        "<english technology name 1>",
        "<русское название 3>",
    ],
)
def test_clean_drops_prompt_hints(placeholder: str):
    """Подсказки из формата промпта: «фраза N» / «phrase N» / «name N», в скобках или без."""
    assert _clean([placeholder, "натрий-ионные аккумуляторы"]) == ["натрий-ионные аккумуляторы"]


def test_clean_keeps_real_phrase_in_angle_brackets():
    """YandexGPT Lite отвечает «<open banking>»: это настоящая фраза, скобки просто снимаем."""
    assert _clean(["<quantum key distribution>", "< homomorphic encryption >", "<a<b>"]) == [
        "quantum key distribution",
        "homomorphic encryption",
    ]

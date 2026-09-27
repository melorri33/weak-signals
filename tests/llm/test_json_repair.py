"""Починка JSON в ответе модели: лишние запятые, переписанная схема, скобки и кавычки — офлайн."""

import json

import pytest
from pydantic import BaseModel

from src.common.config import Settings
from src.llm.client import LLMClient
from src.llm.json_repair import repair_json


class _Card(BaseModel):
    name_ru: str
    facts: list[str]


SCHEMA = json.dumps(_Card.model_json_schema(), ensure_ascii=False)


def test_valid_answer_is_untouched():
    raw = '{"name_ru": "Город", "facts": ["a, ]"]}'
    assert repair_json(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        '{"name_ru": "Город", "facts": ["первый", "второй",]}',
        '{"name_ru": "Город", "facts": ["первый", "второй"],\n}',
    ],
)
def test_trailing_comma_is_removed(raw: str):
    assert _Card.model_validate_json(repair_json(raw)).facts == ["первый", "второй"]


@pytest.mark.parametrize("between", ["\n", " }", "},\n"])
def test_schema_echoed_before_answer_is_dropped(between: str):
    raw = SCHEMA + between + '{"name_ru": "Город", "facts": ["первый"]}'
    assert _Card.model_validate_json(repair_json(raw)).name_ru == "Город"


def test_truncated_answer_is_not_invented():
    raw = SCHEMA + '{"name_ru": "Город", "facts": ["перв'
    assert repair_json(raw) == raw  # недописанное не достраиваем — пусть модель переделает


class _Note(BaseModel):
    name_ru: str
    items: list[dict[str, str]] = []


# Огрехи разметки из ответов GigaChat 2 Pro 27.09: текст на месте, ломаются скобки и кавычки вокруг.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"name_ru": "Город", "items": [{"b": "текст."}]]}', {"name_ru": "Город", "items": [{"b": "текст."}]}),
        ('{"name_ru": "Город", "items": [{"b": "текст."}])}', {"name_ru": "Город", "items": [{"b": "текст."}]}),
        ('{"name_ru": "Город", "items": [{"b": "текст.")}]}', {"name_ru": "Город", "items": [{"b": "текст."}]}),
        ('{"name_ru": "Город", "items": [{"b": "текст."}])', {"name_ru": "Город", "items": [{"b": "текст."}]}),
        ('{"name_ru": "Сорт "Альфа" для полей"}', {"name_ru": 'Сорт "Альфа" для полей'}),
        ('{"name_ru": "Город."", "items": []}', {"name_ru": "Город.", "items": []}),
        (
            '{"name_ru": "Город", "items": [{"b": "раз", {"b": "два"}]}',
            {"name_ru": "Город", "items": [{"b": "раз"}, {"b": "два"}]},
        ),  # noqa: E501
        ('{"name_ru": "Город, где\nпоют"}}', {"name_ru": "Город, где\nпоют"}),
    ],
)
def test_broken_markup_around_complete_text_is_repaired(raw: str, expected: dict):
    assert json.loads(repair_json(raw)) == expected


def test_unclosed_string_loses_bracket_junk_not_text():
    raw = '{"name_ru": "Город", "items": [{"b": "текст.}])}'
    assert _Note.model_validate_json(repair_json(raw)).items == [{"b": "текст."}]


def test_answer_cut_mid_text_is_not_completed():
    raw = '{"name_ru": "Город", "items": [{"b": "обрезанный текс'
    assert repair_json(raw) == raw


class _Backend:
    model = "fake"
    provider = "fake"

    def __init__(self, answers: list[str]):
        self.answers = answers
        self.calls = 0

    async def chat(self, messages, json_schema, max_tokens, timeout_s) -> str:
        self.calls += 1
        return self.answers.pop(0)


async def test_repaired_answer_saves_the_retry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.llm.client.get_settings", lambda: Settings(llm_cache=False))
    backend = _Backend(['{"name_ru": "Город", "facts": ["первый",]}'])
    client = LLMClient(model=backend.model, provider=backend.provider, backend=backend)

    card = await client.ask_json(step="make_card", prompt="тест", schema=_Card)

    assert card.facts == ["первый"] and backend.calls == 1


async def test_unrepairable_answer_still_gets_a_retry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.llm.client.get_settings", lambda: Settings(llm_cache=False))
    backend = _Backend(['{"name_ru": "Город"', '{"name_ru": "Город", "facts": []}'])
    client = LLMClient(model=backend.model, provider=backend.provider, backend=backend)

    card = await client.ask_json(step="make_card", prompt="тест", schema=_Card)

    assert card.name_ru == "Город" and backend.calls == 2

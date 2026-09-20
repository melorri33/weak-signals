"""Офлайн-тесты выделения кандидатов: ответ LLM разбирается и проверяется, отказ не валит шаг."""

import json

import pytest
from pydantic import BaseModel

from src.common.schemas import Candidate, Document, SourceType
from src.llm.client import LLMError
from src.pipeline.candidates import extract_candidates, warn_on_long_names


def _doc(doc_id: str, title: str, source_type: SourceType = SourceType.PAPER) -> Document:
    return Document(
        id=doc_id,
        source="openalex",
        source_type=source_type,
        title=title,
        url=f"https://example.org/{doc_id}",
    )


class _FakeClient:
    """Отдаёт заранее заготовленные ответы по очереди и запоминает промпты."""

    def __init__(self, answers: list[dict | Exception]):
        self._answers = list(answers)
        self.prompts: list[str] = []

    async def ask_json(self, step: str, prompt: str, schema: type[BaseModel], max_tokens: int = 0) -> BaseModel:
        self.prompts.append(prompt)
        answer = self._answers.pop(0) if self._answers else {"candidates": []}
        if isinstance(answer, Exception):
            raise answer
        return schema.model_validate_json(json.dumps(answer))


async def test_llm_names_the_technology_not_the_headline():
    """Главное отличие от заглушки: кандидат — это технология, а не кусок заголовка новости."""
    docs = [
        _doc("a", "Startup raises $12M to run language models on phones", SourceType.NEWS),
        _doc("b", "On-device inference for small language models"),
    ]
    client = _FakeClient(
        [
            {
                "candidates": [
                    {
                        "name": "on-device inference",
                        "name_ru": "инференс на устройстве",
                        "aliases": ["edge inference"],
                        "document_ids": ["d1", "d2"],
                    }
                ]
            }
        ]
    )

    candidates = await extract_candidates(docs, client=client)

    assert [c.name for c in candidates] == ["on-device inference"]
    assert candidates[0].id == "on-device-inference"
    assert candidates[0].name_ru == "инференс на устройстве"
    assert sorted(candidates[0].document_ids) == ["a", "b"]


async def test_same_technology_from_two_batches_is_merged():
    docs = [_doc(str(i), f"Работа {i}") for i in range(14)]  # больше одной пачки
    answer = {"candidates": [{"name": "quantum sensing", "document_ids": ["d1"]}]}
    client = _FakeClient([answer, answer])

    candidates = await extract_candidates(docs, client=client)

    assert len(candidates) == 1
    assert len(candidates[0].document_ids) == 2  # по одному документу из каждой пачки
    assert len(client.prompts) == 2


async def test_invented_document_ids_are_dropped():
    """Модель сослалась на документ, которого в пачке не было: карточку по нему не собрать — выкидываем."""
    docs = [_doc("a", "Работа")]
    client = _FakeClient(
        [
            {
                "candidates": [
                    {"name": "invented technology", "document_ids": ["d99"]},
                    {"name": "quantum sensing", "document_ids": ["d1"]},
                ]
            }
        ]
    )

    candidates = await extract_candidates(docs, client=client)

    assert [c.name for c in candidates] == ["quantum sensing"]


async def test_too_broad_names_are_dropped():
    docs = [_doc("a", "Работа")]
    client = _FakeClient(
        [
            {
                "candidates": [
                    {"name": "artificial intelligence", "document_ids": ["d1"]},
                    {"name": "on-device inference", "document_ids": ["d1"]},
                ]
            }
        ]
    )

    candidates = await extract_candidates(docs, client=client)

    assert [c.name for c in candidates] == ["on-device inference"]


async def test_news_go_to_the_model_first():
    """71% источников датасета — техноновости, поэтому в пачку они должны попадать раньше науки."""
    docs = [_doc(str(i), f"Статья {i}") for i in range(12)]
    docs.append(_doc("news", "Новость про стартап", SourceType.NEWS))
    client = _FakeClient([{"candidates": [{"name": "quantum sensing", "document_ids": ["d1"]}]}])

    await extract_candidates(docs, client=client)

    assert "Новость про стартап" in client.prompts[0]


async def test_falls_back_to_titles_when_llm_is_down():
    """LLM недоступна — шаг не падает: названия режутся из заголовков, прогон продолжается."""
    docs = [
        _doc("a", "Zero-knowledge KYC verification for cross-border payments"),
        _doc("b", "Zero-knowledge KYC verification: benchmarks"),
    ]
    client = _FakeClient([LLMError("Ollama не ответила")])

    candidates = await extract_candidates(docs, client=client)

    assert [c.id for c in candidates] == ["zero-knowledge-kyc-verification"]
    assert candidates[0].document_ids == ["a", "b"]


async def test_no_documents_no_candidates():
    assert await extract_candidates([]) == []


def test_long_name_is_reported(caplog: pytest.LogCaptureFixture):
    """Длинное название: статистику точной фразой по нему не найти — предупреждаем, но кандидата не теряем."""
    candidates = [
        Candidate(id="onboarding", name="Защита онбординга от инъекционных атак и дипфейков"),
        Candidate(id="na-ion", name="sodium-ion batteries"),
    ]

    with caplog.at_level("WARNING", logger="src.pipeline.candidates"):
        returned = warn_on_long_names(candidates)

    assert returned == candidates, "кандидатов не выбрасываем, только предупреждаем"
    assert "Названия длиннее" in caplog.text
    assert "Защита онбординга" in caplog.text
    assert "sodium-ion batteries" not in caplog.text


def test_short_names_do_not_warn(caplog: pytest.LogCaptureFixture):
    with caplog.at_level("WARNING", logger="src.pipeline.candidates"):
        warn_on_long_names([Candidate(id="na-ion", name="sodium-ion batteries")])

    assert caplog.text == ""


async def test_all_answers_unusable_falls_back_to_titles():
    """Модель ответила, но брать нечего — лучше грубые названия из заголовков, чем пустой шаг."""
    docs = [_doc("a", "Sodium-ion batteries for grid storage")]
    client = _FakeClient([{"candidates": [{"name": "blockchain", "document_ids": ["d1"]}]}])

    candidates = await extract_candidates(docs, client=client)

    assert [c.name for c in candidates] == ["Sodium-ion batteries grid"]

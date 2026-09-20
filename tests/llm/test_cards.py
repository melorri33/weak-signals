"""Офлайн-тесты карточек: источники берутся только из переданных документов (требование ТЗ),
тексты пишет модель, а машинные резюме помечаются."""

import json

import pytest
from pydantic import BaseModel

from src.common.schemas import Document, ScoredCandidate, SourceType, TrustLevel
from src.llm.cards import MAX_SOURCES, NO_TEXT, NoSourcesError, keep_known_documents, make_card
from src.llm.client import LLMError

FULL_ANSWER = {
    "description": "Инференс выполняется на самом устройстве, без обращения к облаку.",
    "advantage": "Ниже задержка и не уходят данные пользователя.",
    "case_example": "Пробуют в смартфонах и промышленных датчиках.",
    "why_weak_signal": "Пока единичные внедрения и в основном препринты.",
    "used_document_ids": ["a"],
    "source_summaries": [{"document_id": "a", "ru_summary": "Разбор запуска модели на устройстве."}],
}


def _doc(doc_id: str, source_type: SourceType = SourceType.PAPER) -> Document:
    return Document(
        id=doc_id,
        source="openalex",
        source_type=source_type,
        title=f"Работа {doc_id}",
        url=f"https://example.org/{doc_id}",
        language="en",
        trust=TrustLevel.HIGH,
    )


def _scored() -> ScoredCandidate:
    return ScoredCandidate(candidate_id="on-device", name="on-device inference", score=0.8)


class _FakeClient:
    def __init__(self, answer: dict | Exception = None):
        self._answer = answer if answer is not None else FULL_ANSWER
        self.prompts: list[str] = []

    async def ask_json(self, step: str, prompt: str, schema: type[BaseModel], max_tokens: int = 0) -> BaseModel:
        self.prompts.append(prompt)
        if isinstance(self._answer, Exception):
            raise self._answer
        return schema.model_validate_json(json.dumps(self._answer))


async def test_card_sources_from_documents():
    docs = [_doc("a"), _doc("b", SourceType.NEWS)]
    card = await make_card(_scored(), docs, name_ru="Инференс на устройстве", client=_FakeClient())

    assert card.name_ru == "Инференс на устройстве"
    assert [s.document_id for s in card.sources] == ["a", "b"]
    assert [s.url for s in card.sources] == ["https://example.org/a", "https://example.org/b"]
    assert card.sources[0].trust == TrustLevel.HIGH
    assert card.sources[1].source_type == SourceType.NEWS


async def test_card_texts_come_from_the_model():
    card = await make_card(_scored(), [_doc("a")], client=_FakeClient())

    assert card.description == FULL_ANSWER["description"]
    assert card.advantage == FULL_ANSWER["advantage"]
    assert card.case_example == FULL_ANSWER["case_example"]
    assert card.why_weak_signal == FULL_ANSWER["why_weak_signal"]


async def test_generated_summary_is_marked():
    """ТЗ: сгенерированное резюме должно быть видно пользователю как машинное."""
    card = await make_card(_scored(), [_doc("a"), _doc("b")], client=_FakeClient())

    with_summary = next(s for s in card.sources if s.document_id == "a")
    without_summary = next(s for s in card.sources if s.document_id == "b")

    assert with_summary.ru_summary == FULL_ANSWER["source_summaries"][0]["ru_summary"]
    assert with_summary.summary_is_generated is True
    assert without_summary.ru_summary is None
    assert without_summary.summary_is_generated is False


async def test_summary_for_unknown_document_is_dropped():
    answer = {**FULL_ANSWER, "source_summaries": [{"document_id": "выдумка", "ru_summary": "текст"}]}
    card = await make_card(_scored(), [_doc("a")], client=_FakeClient(answer))

    assert card.sources[0].ru_summary is None


async def test_card_survives_llm_failure():
    """Модель недоступна — сигнал не теряем: карточка собирается с честной пометкой."""
    card = await make_card(_scored(), [_doc("a")], client=_FakeClient(LLMError("Ollama не ответила")))

    assert card.description == NO_TEXT
    assert card.sources[0].document_id == "a"
    assert card.sources[0].summary_is_generated is False


async def test_prompt_gets_only_this_candidate_documents():
    client = _FakeClient()
    await make_card(_scored(), [_doc("a")], client=client)

    assert "Работа a" in client.prompts[0]
    assert "on-device inference" in client.prompts[0]


async def test_card_limits_sources():
    card = await make_card(_scored(), [_doc(str(i)) for i in range(10)], client=_FakeClient())
    assert len(card.sources) == MAX_SOURCES


async def test_card_without_documents_is_rejected():
    with pytest.raises(NoSourcesError):
        await make_card(_scored(), [], client=_FakeClient())


def test_invented_document_ids_are_dropped():
    docs = [_doc("real")]
    kept = keep_known_documents(["real", "openalex:W999-выдумка", "real"], docs)
    assert [d.id for d in kept] == ["real"]

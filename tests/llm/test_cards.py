"""Офлайн-тесты карточек: источники берутся только из переданных документов (требование ТЗ)."""

import pytest

from src.common.schemas import Document, ScoredCandidate, SourceType, TrustLevel
from src.llm.cards import MAX_SOURCES, NoSourcesError, keep_known_documents, make_card


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
    return ScoredCandidate(candidate_id="zk-kyc", name="Zero-knowledge KYC", score=0.8)


async def test_card_sources_from_documents():
    docs = [_doc("a"), _doc("b", SourceType.NEWS)]
    card = await make_card(_scored(), docs, name_ru="KYC с нулевым разглашением")

    assert card.name_ru == "KYC с нулевым разглашением"
    assert [s.document_id for s in card.sources] == ["a", "b"]
    assert [s.url for s in card.sources] == ["https://example.org/a", "https://example.org/b"]
    assert card.sources[0].trust == TrustLevel.HIGH
    assert card.sources[1].source_type == SourceType.NEWS


async def test_card_limits_sources():
    card = await make_card(_scored(), [_doc(str(i)) for i in range(10)])
    assert len(card.sources) == MAX_SOURCES


async def test_card_without_documents_is_rejected():
    with pytest.raises(NoSourcesError):
        await make_card(_scored(), [])


def test_invented_document_ids_are_dropped():
    docs = [_doc("real")]
    kept = keep_known_documents(["real", "openalex:W999-выдумка", "real"], docs)
    assert [d.id for d in kept] == ["real"]

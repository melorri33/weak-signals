"""Офлайн-тесты заглушки выделения кандидатов."""

from src.common.schemas import Document, SourceType
from src.pipeline.candidates import extract_candidates


def _doc(doc_id: str, title: str) -> Document:
    return Document(
        id=doc_id,
        source="openalex",
        source_type=SourceType.PAPER,
        title=title,
        url=f"https://example.org/{doc_id}",
    )


async def test_same_technology_collects_documents():
    docs = [
        _doc("a", "Zero-knowledge KYC verification for cross-border payments"),
        _doc("b", "Zero-knowledge KYC verification: benchmarks"),
        _doc("c", "Federated credit scoring across small banks"),
    ]

    candidates = await extract_candidates(docs)

    assert [c.id for c in candidates] == ["zero-knowledge-kyc-verification", "federated-credit-scoring"]
    assert candidates[0].document_ids == ["a", "b"]  # одинаковое начало названия — один кандидат
    assert candidates[0].name == "Zero-knowledge KYC verification"


async def test_no_documents_no_candidates():
    assert await extract_candidates([]) == []

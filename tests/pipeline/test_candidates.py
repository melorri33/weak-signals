"""Офлайн-тесты заглушки выделения кандидатов."""

import pytest

from src.common.schemas import Candidate, Document, SourceType
from src.pipeline.candidates import extract_candidates, warn_on_long_names


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


def test_long_name_is_reported(caplog: pytest.LogCaptureFixture):
    """Длинное название: статистику точной фразой по нему не найти — предупреждаем, но кандидата не теряем.

    Заглушка обрезает названия до трёх слов сама, поэтому проверка нужна для настоящей версии на LLM.
    """
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

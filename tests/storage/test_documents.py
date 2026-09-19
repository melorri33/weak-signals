"""save_documents/get_documents на изолированной SQLite-базе (в проде — PostgreSQL, но SQL один и тот же)."""

from __future__ import annotations

import pytest

from src.common.config import Settings
from src.common.schemas import Document, SourceType
from src.storage import db, documents


def _doc(doc_id: str, title: str = "документ") -> Document:
    return Document(
        id=doc_id, source="openalex", source_type=SourceType.PAPER, title=title, url=f"https://example.org/{doc_id}"
    )


def test_save_and_get_roundtrip():
    documents.save_documents([_doc("openalex:1"), _doc("openalex:2")])

    result = documents.get_documents(["openalex:1", "openalex:2"])

    assert {d.id for d in result} == {"openalex:1", "openalex:2"}


def test_get_documents_missing_id_is_skipped():
    documents.save_documents([_doc("openalex:1")])

    result = documents.get_documents(["openalex:1", "openalex:nonexistent"])

    assert [d.id for d in result] == ["openalex:1"]


def test_get_documents_empty_list_returns_empty():
    assert documents.get_documents([]) == []


def test_save_documents_empty_list_does_not_fail():
    documents.save_documents([])


def test_save_documents_upserts_by_id():
    documents.save_documents([_doc("openalex:1", title="старое название")])
    documents.save_documents([_doc("openalex:1", title="новое название")])

    result = documents.get_documents(["openalex:1"])

    assert len(result) == 1
    assert result[0].title == "новое название"


def test_unavailable_database_does_not_raise(monkeypatch: pytest.MonkeyPatch):
    """README: конвейер должен работать и без базы — сбой БД не должен ронять вызывающий код."""
    monkeypatch.setattr(db, "get_settings", lambda: Settings(database_url="postgresql+psycopg://x:x@localhost:1/x"))
    db.get_engine.cache_clear()

    documents.save_documents([_doc("openalex:1")])  # не должно бросить исключение

    assert documents.get_documents(["openalex:1"]) == []

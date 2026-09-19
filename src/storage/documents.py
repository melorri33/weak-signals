"""save_documents/get_documents: хранение найденных документов (storage.save_documents/get_documents)."""

from __future__ import annotations

from sqlalchemy import select

from src.common.schemas import Document
from src.storage.db import db_unavailable_ok, get_session, init_db
from src.storage.models import DocumentRow


@db_unavailable_ok(default_factory=lambda: None)
def save_documents(docs: list[Document]) -> None:
    """Сохранить документы, без дублей по id (повторное сохранение обновляет запись)."""
    if not docs:
        return
    init_db()
    with get_session() as session:
        for doc in docs:
            row = session.get(DocumentRow, doc.id)
            if row is None:
                row = DocumentRow(id=doc.id)
                session.add(row)
            row.source = doc.source
            row.source_type = doc.source_type.value
            row.title = doc.title
            row.url = doc.url
            row.published = doc.published
            row.language = doc.language
            row.fetched_at = doc.fetched_at
            row.raw = doc.model_dump(mode="json")
        session.commit()


@db_unavailable_ok(default_factory=list)
def get_documents(ids: list[str]) -> list[Document]:
    """Документы по списку id (порядок не гарантирован)."""
    if not ids:
        return []
    init_db()
    with get_session() as session:
        rows = session.execute(select(DocumentRow).where(DocumentRow.id.in_(ids))).scalars().all()
    return [Document.model_validate(row.raw) for row in rows]

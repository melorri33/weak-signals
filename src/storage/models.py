"""Черновик таблиц PostgreSQL (SQLAlchemy 2.0).

Пока только схема — save_documents/get_documents/save_search_result/get_search_result
появятся отдельным PR (дни 3-4), когда будет на чём проверять дедупликацию и кэш прогонов.
Обе таблицы хранят объект целиком в `raw` (JSON) — не теряем поля из схемы, пока она меняется,
плюс несколько колонок для фильтров и индексов.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    """Один документ (src.common.schemas.Document)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # '<source>:<внешний id>'
    source: Mapped[str] = mapped_column(String, index=True)
    source_type: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String)
    published: Mapped[date | None] = mapped_column(Date, nullable=True)
    language: Mapped[str] = mapped_column(String)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw: Mapped[dict] = mapped_column(JSON)  # Document.model_dump(mode="json") целиком


class SearchResultRow(Base):
    """Результат прогона (src.common.schemas.SearchResult), run_id — ключ."""

    __tablename__ = "search_results"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    query: Mapped[str] = mapped_column(Text, index=True)
    status: Mapped[str] = mapped_column(String, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw: Mapped[dict] = mapped_column(JSON)  # SearchResult.model_dump(mode="json") целиком

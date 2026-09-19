"""save_search_result/get_search_result: результат прогона по run_id."""

from __future__ import annotations

from src.common.schemas import SearchResult
from src.storage.db import db_unavailable_ok, get_session, init_db
from src.storage.models import SearchResultRow


@db_unavailable_ok(default_factory=lambda: None)
def save_search_result(result: SearchResult) -> None:
    """Сохранить промежуточный или итоговый результат прогона (перезаписывает по run_id)."""
    init_db()
    with get_session() as session:
        row = session.get(SearchResultRow, result.run_id)
        if row is None:
            row = SearchResultRow(run_id=result.run_id)
            session.add(row)
        row.query = result.query
        row.status = result.status
        row.started_at = result.started_at
        row.raw = result.model_dump(mode="json")
        session.commit()


@db_unavailable_ok(default_factory=lambda: None)
def get_search_result(run_id: str) -> SearchResult | None:
    init_db()
    with get_session() as session:
        row = session.get(SearchResultRow, run_id)
    return SearchResult.model_validate(row.raw) if row else None

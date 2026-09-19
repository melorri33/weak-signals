"""save_search_result/get_search_result на изолированной SQLite-базе."""

from __future__ import annotations

import pytest

from src.common.config import Settings
from src.common.schemas import SearchResult
from src.storage import db, search_results


def test_save_and_get_roundtrip():
    result = SearchResult(run_id="run-1", query="финтех", status="done")

    search_results.save_search_result(result)
    loaded = search_results.get_search_result("run-1")

    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.query == "финтех"
    assert loaded.status == "done"


def test_get_search_result_missing_returns_none():
    assert search_results.get_search_result("no-such-run") is None


def test_save_search_result_overwrites_by_run_id():
    search_results.save_search_result(SearchResult(run_id="run-1", query="a", status="running"))
    search_results.save_search_result(SearchResult(run_id="run-1", query="a", status="done"))

    loaded = search_results.get_search_result("run-1")

    assert loaded.status == "done"


def test_unavailable_database_does_not_raise(monkeypatch: pytest.MonkeyPatch):
    """README: конвейер должен работать и без базы — сбой БД не должен ронять вызывающий код."""
    monkeypatch.setattr(db, "get_settings", lambda: Settings(database_url="postgresql+psycopg://x:x@localhost:1/x"))
    db.get_engine.cache_clear()

    search_results.save_search_result(SearchResult(run_id="run-1", query="a", status="done"))  # не должно бросить

    assert search_results.get_search_result("run-1") is None

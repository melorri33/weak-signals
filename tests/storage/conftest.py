"""Общее для тестов хранилища: изолированная SQLite-база на файл во временной папке."""

from __future__ import annotations

import pytest

from src.common.config import Settings
from src.storage import db


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch: pytest.MonkeyPatch, tmp_path):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "get_settings", lambda: Settings(database_url=f"sqlite:///{db_path}"))
    db.get_engine.cache_clear()
    db.forget_db_state()
    yield
    db.get_engine.cache_clear()
    db.forget_db_state()

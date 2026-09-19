"""Поведение при недоступной базе: конвейер работает без неё и не ждёт таймаут на каждом шаге."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from src.storage import db


@pytest.fixture
def failing_call():
    """Функция, которая всегда падает как недоступная база, и счётчик её вызовов."""
    calls = []

    @db.db_unavailable_ok(default_factory=list)
    def save(value: str) -> list[str]:
        calls.append(value)
        raise OperationalError("SELECT 1", {}, Exception("connection timeout expired"))

    return save, calls


def test_db_unavailable_returns_default(failing_call):
    save, calls = failing_call

    assert save("первый") == []
    assert calls == ["первый"]


def test_db_is_tried_once_per_run(failing_call):
    """Сохранение вызывается на каждом шаге конвейера: после первой неудачи в базу больше не ходим."""
    save, calls = failing_call

    for value in ("первый", "второй", "третий"):
        assert save(value) == []

    assert calls == ["первый"]


def test_forget_db_state_allows_retry(failing_call):
    save, calls = failing_call

    save("первый")
    db.forget_db_state()
    save("второй")

    assert calls == ["первый", "второй"]

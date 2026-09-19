"""Подключение к базе. save_documents/get_documents/save_search_result/get_search_result — в
src/storage/documents.py и search_results.py."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, wraps
from typing import TypeVar

from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import get_settings
from src.common.logs import get_logger
from src.storage.models import Base

log = get_logger(__name__)

T = TypeVar("T")

# Без этого таймаута недоступный Postgres не падает быстро, а виснет на сетевом таймауте ОС —
# на практике это оказалось ~50 минут на весь офлайн-прогон тестов вместо секунд. SQLite (тесты)
# такой параметр не понимает, поэтому передаём его только для postgres.
DB_CONNECT_TIMEOUT_S = 2


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    connect_args = {"connect_timeout": DB_CONNECT_TIMEOUT_S} if url.startswith("postgresql") else {}
    return create_engine(url, connect_args=connect_args)


def init_db() -> None:
    """Создать таблицы, если их ещё нет."""
    Base.metadata.create_all(get_engine())


def get_session() -> Session:
    return sessionmaker(bind=get_engine())()


# База недоступна — не пробуем снова до конца прогона: сохранение вызывается на каждом шаге конвейера,
# и каждая попытка ждала бы DB_CONNECT_TIMEOUT_S. На прогоне без Postgres это давало +30 секунд впустую.
_db_down = False


def forget_db_state() -> None:
    """Забыть, что база была недоступна (тесты и смена настроек подключения)."""
    global _db_down
    _db_down = False


def db_unavailable_ok(default_factory: Callable[[], T]) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """База недоступна — по README конвейер должен работать и без неё: лог и результат `default_factory()`.

    Фабрика, а не готовое значение — иначе один и тот же список/словарь расшарился бы между вызовами.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> T:
            global _db_down
            if _db_down:
                return default_factory()
            try:
                return fn(*args, **kwargs)
            except SQLAlchemyError as exc:
                log.warning("база недоступна (%s): %s — дальше работаем без неё", fn.__name__, exc)
                _db_down = True
                return default_factory()

        return wrapper

    return decorator

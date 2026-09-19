"""Подключение к PostgreSQL. save_documents/get_documents — отдельным PR (дни 3-4)."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import get_settings
from src.storage.models import Base


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url)


def init_db() -> None:
    """Создать таблицы, если их ещё нет."""
    Base.metadata.create_all(get_engine())


def get_session() -> Session:
    return sessionmaker(bind=get_engine())()

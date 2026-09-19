"""Хранилище: PostgreSQL и кэш (owner: Данные). Контракты — в src/common/schemas.py."""

from src.storage.documents import get_documents, save_documents
from src.storage.search_results import get_search_result, save_search_result

__all__ = ["get_documents", "get_search_result", "save_documents", "save_search_result"]

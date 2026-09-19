"""Склейка документов из разных источников по DOI, а если его нет — по нормализованному заголовку."""

from __future__ import annotations

import re

from src.common.schemas import Document

_DOI_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


def _dedupe_key(doc: Document) -> str:
    if "doi.org/" in doc.url.lower():
        return "doi:" + _DOI_PREFIX.sub("", doc.url).lower()
    normalized_title = re.sub(r"[^a-zа-я0-9]+", " ", doc.title.lower()).strip()
    return "title:" + normalized_title


def dedupe(docs: list[Document]) -> list[Document]:
    """Первый встреченный документ побеждает — порядок источников задаёт приоритет."""
    seen: dict[str, Document] = {}
    for doc in docs:
        key = _dedupe_key(doc)
        if key not in seen:
            seen[key] = doc
    return list(seen.values())

"""Общее для офлайн-тестов."""

import pytest


@pytest.fixture(autouse=True)
def _no_name_model(monkeypatch):
    """Модель названий тянет bge-m3 (2 ГБ, полминуты загрузки) — в офлайн-тестах её нет.

    Тесты самой модели названий (tests/model/test_names.py) подменяют name_scores своей заглушкой.
    """
    from src.model import names

    monkeypatch.setattr(names, "name_scores", lambda _names, **_kwargs: None)


@pytest.fixture(autouse=True)
def _no_embeddings_for_ranking(monkeypatch):
    """Близость к теме и почти-дубли считают эмбеддинги bge-m3 — в офлайн-тестах без них.

    Тесты самих поправок (tests/model/test_relevance.py) подменяют _embed своей заглушкой.
    """
    from src.model import relevance

    monkeypatch.setattr(relevance, "_embed", lambda _texts, step: None)

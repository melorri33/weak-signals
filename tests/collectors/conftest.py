"""Общее для тестов сборщиков: настройки без сети и доступ к сохранённым ответам API."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.collectors import arxiv, cache
from src.collectors.http import RateLimiter
from src.common.config import Settings

RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


def raw_fixture(name: str) -> str:
    return (RAW / name).read_text(encoding="utf-8")


@pytest.fixture
def settings() -> Settings:
    return Settings(source_timeout_s=5.0, collect_budget_s=10.0, cache_ttl_days=7, contact_email="test@example.org")


@pytest.fixture(autouse=True)
def _no_arxiv_rate_limit(monkeypatch: pytest.MonkeyPatch):
    """В проде arXiv просит паузу 3 с между запросами — в тестах она бы только всё замедляла."""
    monkeypatch.setattr(arxiv, "_rate_limiter", RateLimiter(interval_s=0))


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Кэш пишется в data/cache/ — в тестах уводим его во временную папку, чтобы не мусорить и не течь между тестами."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")

"""Файловый кэш: ключ по источнику+термину, TTL из настроек."""

from __future__ import annotations

import time

from src.collectors import cache


def test_roundtrip():
    assert cache.get("openalex_years", "quantum sensing") is None

    cache.set("openalex_years", "quantum sensing", value={"2025": 10})

    assert cache.get("openalex_years", "quantum sensing") == {"2025": 10}


def test_different_terms_and_sources_do_not_collide():
    cache.set("openalex_years", "a", value=1)
    cache.set("openalex_years", "b", value=2)
    cache.set("hn_news", "a", value=3)

    assert cache.get("openalex_years", "a") == 1
    assert cache.get("openalex_years", "b") == 2
    assert cache.get("hn_news", "a") == 3


def test_expired_entry_is_ignored(monkeypatch):
    from src.common.config import Settings

    monkeypatch.setattr(cache, "get_settings", lambda: Settings(cache_ttl_days=0))
    cache.set("openalex_years", "term", value=42)
    time.sleep(0.01)

    assert cache.get("openalex_years", "term") is None

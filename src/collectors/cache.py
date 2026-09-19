"""Файловый кэш ответов внешних API. Ключ = источник + термин (+ интервал), TTL = CACHE_TTL_DAYS.

Нужен, чтобы демо и повторные прогоны не жгли дневной бюджет OpenAlex и не упирались в лимиты
Hacker News/Википедии. Кэш кладём в data/cache/ — .gitignore не коммитит содержимое data/.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.common.config import get_settings

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _path(source: str, *key_parts: str) -> Path:
    raw = "|".join((source, *key_parts))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return CACHE_DIR / source / f"{digest}.json"


def get(source: str, *key_parts: str) -> Any | None:
    """Значение из кэша или None, если его нет или оно устарело."""
    path = _path(source, *key_parts)
    if not path.exists():
        return None
    ttl_s = get_settings().cache_ttl_days * 86400
    if time.time() - path.stat().st_mtime > ttl_s:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def set(source: str, *key_parts: str, value: Any) -> None:
    """Положить значение в кэш (перезаписывает время жизни)."""
    path = _path(source, *key_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

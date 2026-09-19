"""Уровень доверия к источнику: `level_for(url, source_type) -> TrustLevel`.

Сначала — уровень по типу источника, затем уточнение по домену из config/trust_domains.yaml.
Неизвестный домен оставляет уровень по типу. Правило ТЗ «соцсети, блоги, пресс-релизы не могут быть
единственным основанием» проверяется не здесь, а в правилах отсева шума (src/filters).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml

from src.common.schemas import SourceType, TrustLevel

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "trust_domains.yaml"
_ORDER = (TrustLevel.HIGH, TrustLevel.MEDIUM, TrustLevel.LOW)


@lru_cache(maxsize=1)
def _config() -> tuple[dict[SourceType, TrustLevel], list[tuple[str, TrustLevel]]]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    by_type = {SourceType(k): TrustLevel(v) for k, v in raw["by_source_type"].items()}
    domains = [(d.lower(), level) for level in _ORDER for d in raw["domains"].get(level.value, [])]
    return by_type, domains


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def _matches(host: str, pattern: str) -> bool:
    """«.gov» — любой домен на .gov; «nature.com» — сам домен и его поддомены, но не «notnature.com»."""
    if pattern.startswith("."):
        return host.endswith(pattern)
    return host == pattern or host.endswith("." + pattern)


def domain_level(url: str) -> TrustLevel | None:
    """Уровень по домену или None, если домен в конфиге не описан."""
    host = domain_of(url)
    if not host:
        return None
    _, domains = _config()
    return next((level for pattern, level in domains if _matches(host, pattern)), None)


def level_for(url: str, source_type: SourceType) -> TrustLevel:
    """Уровень доверия документа: домен из конфига важнее типа, неизвестный домен — по типу."""
    by_type, _ = _config()
    return domain_level(url) or by_type.get(SourceType(source_type), TrustLevel.LOW)

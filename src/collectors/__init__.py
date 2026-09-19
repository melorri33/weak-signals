"""Сборщики открытых источников (owner: Данные). Контракты — в src/common/schemas.py."""

from src.collectors.collect import collect
from src.collectors.term_stats import term_stats

__all__ = ["collect", "term_stats"]

"""Правила отсева: `apply(candidate, features) -> FilterDecision`.

Отсекают до модели только очевидное — зрелое, хайп и шум — и всегда объясняют причину по-русски с цифрами
(причина показывается во вкладке «Отсеяно», требование ТЗ). Пороги — config/filter_rules.yaml.
Признак без данных (None) правило не включает: отсутствие данных — не повод исключать.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.common.schemas import Candidate, CandidateFeatures, FilterDecision

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "filter_rules.yaml"


@lru_cache(maxsize=1)
def rules() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _num(x: float | int) -> str:
    return f"{int(x):,}".replace(",", " ")


def _noise(f: CandidateFeatures, cfg: dict[str, Any]) -> str | None:
    rated, trusted = f.extra.get("docs_rated"), f.extra.get("docs_trusted")
    if cfg.get("require_trusted_source") and rated and trusted == 0:
        return (
            f"Недостаточно подтверждений: все {_num(rated)} источника(ов) — блоги, соцсети или пресс-релизы, "
            "независимых источников нет"
        )
    return None


def _mature(f: CandidateFeatures, cfg: dict[str, Any]) -> str | None:
    reasons = []
    if f.total_pubs is not None and f.total_pubs > cfg["total_pubs_min"]:
        reasons.append(f"{_num(f.total_pubs)} публикаций (порог {_num(cfg['total_pubs_min'])})")
    if cfg.get("has_standard") and f.has_standard:
        reasons.append("есть отраслевой стандарт")
    return "Зрелая технология: " + ", ".join(reasons) if reasons else None


def _hype(f: CandidateFeatures, cfg: dict[str, Any]) -> str | None:
    news, ratio = f.news_total, f.news_to_science_ratio
    if news is None or ratio is None:
        return None
    if news >= cfg["news_total_min"] and ratio >= cfg["news_to_science_ratio_min"]:
        return (
            f"Медийная, а не исследовательская тема: {_num(news)} упоминаний в техмедиа — "
            f"{ratio:.1f} на одну научную работу "
            f"(пороги {_num(cfg['news_total_min'])} и {cfg['news_to_science_ratio_min']})"
        )
    return None


# Порядок важен: шум — жёсткое требование ТЗ, затем зрелое, затем хайп.
_CHECKS = (("noise", _noise), ("mature", _mature), ("hype", _hype))


def apply(candidate: Candidate, features: CandidateFeatures) -> FilterDecision:
    """Исключить ли кандидата и почему. Первое сработавшее правило определяет причину."""
    cfg = rules()
    for code, check in _CHECKS:
        reason = check(features, cfg[code])
        if reason:
            return FilterDecision(
                candidate_id=candidate.id, name=candidate.name, excluded=True, reason_code=code, reason_text=reason
            )
    return FilterDecision(
        candidate_id=candidate.id,
        name=candidate.name,
        excluded=False,
        reason_code="ok",
        reason_text="Правила отсева не сработали — оценку даёт модель",
    )

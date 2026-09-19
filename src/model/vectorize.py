"""CandidateFeatures → числовой вектор модели и русские тексты объяснений.

Общий для обучения (train.py) и инференса (score.py): порядок и смысл признаков задаются только здесь.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from src.common.schemas import CandidateFeatures


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    title_ru: str  # короткое название для отчёта и SHAP-графика


FEATURES: list[FeatureSpec] = [
    FeatureSpec("log_total_pubs", "Публикаций всего (лог)"),
    FeatureSpec("log_pubs_last_year", "Публикаций за последний год (лог)"),
    FeatureSpec("growth_3y", "Рост публикаций за 3 года"),
    FeatureSpec("recent_share", "Доля публикаций за 3 года"),
    FeatureSpec("age_years", "Возраст темы, лет"),
    FeatureSpec("log_news_total", "Упоминаний в техмедиа (лог)"),
    FeatureSpec("log_news_recent", "Упоминаний в техмедиа за 3 года (лог)"),
    FeatureSpec("news_to_science", "Медиа на одну научную работу (лог)"),
    FeatureSpec("has_wikipedia", "Есть статья в Википедии"),
    FeatureSpec("preprint_share", "Доля препринтов"),
]
FEATURE_NAMES = [f.name for f in FEATURES]


def _log(x: float | int | None) -> float | None:
    return math.log1p(x) if x is not None else None


def to_row(f: CandidateFeatures) -> list[float | None]:
    """Вектор признаков в порядке FEATURES. None — пропуск (CatBoost их переживает)."""
    age = date.today().year - f.first_seen_year if f.first_seen_year is not None else None
    ratio = f.news_to_science_ratio
    values = {
        "log_total_pubs": _log(f.total_pubs),
        "log_pubs_last_year": _log(f.extra.get("pubs_last_year")),
        "growth_3y": f.growth_3y,
        "recent_share": f.extra.get("recent_share"),
        "age_years": age,
        "log_news_total": _log(f.news_total),
        "log_news_recent": _log(f.extra.get("news_recent")),
        "news_to_science": _log(ratio * 1000) if ratio is not None else None,
        "has_wikipedia": float(f.has_wikipedia) if f.has_wikipedia is not None else None,
        "preprint_share": f.extra.get("preprint_share"),
    }
    return [values[name] for name in FEATURE_NAMES]


def explain_ru(name: str, f: CandidateFeatures) -> str:
    """Человеческое объяснение признака: «Публикации растут на 60% в год». Знак вклада — в Explanation."""
    last = f.extra.get("pubs_last_year")
    texts = {
        "log_total_pubs": f"Всего публикаций: {_fmt(f.total_pubs)}",
        "log_pubs_last_year": f"Публикаций за последний год: {_fmt(last)}",
        "growth_3y": f"Публикации растут на {round((f.growth_3y or 0) * 100)}% в год",
        "recent_share": f"{round((f.extra.get('recent_share') or 0) * 100)}% публикаций — за последние 3 года",
        "age_years": f"Первые публикации — {f.first_seen_year} г.",
        "log_news_total": f"Упоминаний в техмедиа: {_fmt(f.news_total)}",
        "log_news_recent": f"Упоминаний в техмедиа за 3 года: {_fmt(f.extra.get('news_recent'))}",
        "news_to_science": f"Медиа-упоминаний на одну научную работу: {f.news_to_science_ratio or 0:.2f}",
        "has_wikipedia": "Есть статья в Википедии" if f.has_wikipedia else "Нет статьи в Википедии",
        "preprint_share": f"{round((f.extra.get('preprint_share') or 0) * 100)}% публикаций — препринты",
    }
    return texts[name]


def _fmt(x: float | int | None) -> str:
    return "нет данных" if x is None else f"{int(x):,}".replace(",", " ")

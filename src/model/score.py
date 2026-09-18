"""Скоринг кандидатов: `score(candidates, features) -> list[ScoredCandidate]`.

Основной путь — обученная CatBoost-модель (src/model/artifacts/weak_signal.cbm, собирается `python -m src.model.train`),
объяснения — SHAP-вклады признаков (считает сам CatBoost). Если файла модели нет — прозрачная эвристика
того же вида (вклады в логит), чтобы конвейер работал с нуля.
Каждый вызов пишется в журнал моделей (требование ТЗ).
"""

from __future__ import annotations

import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logs import get_logger, model_timer
from src.common.schemas import Candidate, CandidateFeatures, Explanation, ScoredCandidate
from src.model.vectorize import FEATURE_NAMES, explain_ru, to_row

MODEL_PATH = Path(__file__).parent / "artifacts" / "weak_signal.cbm"
MODEL_NAME = "catboost-weak-signal"
HEURISTIC_NAME = "heuristic-stub-v0"
MODEL_PROVIDER = "local"
TOP_REASONS = 3

log = get_logger(__name__)


def score(candidates: list[Candidate], features: list[CandidateFeatures]) -> list[ScoredCandidate]:
    """Оценить кандидатов. Результат отсортирован по убыванию уверенности.

    Кандидат без признаков получает нейтральные 0.5 без причин.
    """
    by_id = {f.candidate_id: f for f in features}
    model = _load_model()
    name = MODEL_NAME if model is not None else HEURISTIC_NAME
    with model_timer(step="score", model=name, provider=MODEL_PROVIDER):
        if model is not None:
            scored = _score_with_model(model, candidates, by_id)
        else:
            scored = [_score_heuristic(c, by_id.get(c.id)) for c in candidates]
    return sorted(scored, key=lambda s: s.score, reverse=True)


@lru_cache(maxsize=1)
def _load_model():  # -> CatBoostClassifier | None
    if not MODEL_PATH.exists():
        log.warning("Нет файла модели %s — работает эвристика. Обучить: python -m src.model.train", MODEL_PATH)
        return None
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def _score_with_model(model, candidates: list[Candidate], by_id: dict[str, CandidateFeatures]) -> list[ScoredCandidate]:
    known = [c for c in candidates if c.id in by_id]
    out = [ScoredCandidate(candidate_id=c.id, name=c.name, score=0.5) for c in candidates if c.id not in by_id]
    if not known:
        return out
    from catboost import Pool

    X = pd.DataFrame([to_row(by_id[c.id]) for c in known], columns=FEATURE_NAMES, dtype=float)
    proba = model.predict_proba(X)[:, 1]
    shap = model.get_feature_importance(Pool(X), type="ShapValues")[:, :-1]
    for c, p, contrib, row in zip(known, proba, shap, X.itertuples(index=False), strict=True):
        f = by_id[c.id]
        order = np.argsort(-np.abs(contrib))
        reasons = [
            Explanation(
                feature=FEATURE_NAMES[j],
                value=None if pd.isna(row[j]) else round(float(row[j]), 4),
                contribution=round(float(contrib[j]), 4),
                text=explain_ru(FEATURE_NAMES[j], f),
            )
            for j in order[:TOP_REASONS]
            if contrib[j] != 0 and not pd.isna(row[j])
        ]
        out.append(ScoredCandidate(candidate_id=c.id, name=c.name, score=round(float(p), 4), top_reasons=reasons))
    return out


# ---------- Запасной путь: эвристика без обученной модели ----------


def _score_heuristic(candidate: Candidate, f: CandidateFeatures | None) -> ScoredCandidate:
    reasons = _heuristic_reasons(f) if f else []
    logit = sum(r.contribution for r in reasons)
    meaningful = [r for r in reasons if r.contribution != 0]
    top = sorted(meaningful, key=lambda r: abs(r.contribution), reverse=True)[:TOP_REASONS]
    return ScoredCandidate(
        candidate_id=candidate.id,
        name=candidate.name,
        score=round(1 / (1 + math.exp(-logit)), 4),
        top_reasons=top,
    )


def _heuristic_reasons(f: CandidateFeatures) -> list[Explanation]:
    """Вклады признаков: >0 — за слабый сигнал, <0 — против. Нет данных — нет вклада."""
    out: list[Explanation] = []
    if f.growth_3y is not None:
        pct = round(f.growth_3y * 100)
        out.append(
            _expl("growth_3y", f.growth_3y, max(-1.5, min(1.5, f.growth_3y * 2)), f"Публикации растут на {pct}% в год")
        )
    if f.total_pubs is not None:
        c = 0.8 if f.total_pubs < 2000 else (-1.5 if f.total_pubs > 20000 else 0.0)
        out.append(_expl("total_pubs", f.total_pubs, c, f"Всего публикаций: {f.total_pubs:,}".replace(",", " ")))
    if f.first_seen_year is not None:
        age = date.today().year - f.first_seen_year
        c = 0.6 if age <= 7 else (-0.8 if age > 12 else 0.0)
        out.append(_expl("first_seen_year", f.first_seen_year, c, f"Первые публикации — {f.first_seen_year} г."))
    if f.news_to_science_ratio is not None:
        c = -1.2 if f.news_to_science_ratio > 5 else 0.3
        out.append(
            _expl(
                "news_to_science_ratio",
                f.news_to_science_ratio,
                c,
                f"Новостей на одну научную работу: {f.news_to_science_ratio:.1f}",
            )
        )
    if f.has_standard:
        out.append(_expl("has_standard", "да", -1.0, "Есть стандарт — технология уже сформировалась"))
    if f.has_wikipedia:
        out.append(_expl("has_wikipedia", "да", -0.5, "Есть статья в Википедии"))
    return out


def _expl(feature: str, value: float | str, contribution: float, text: str) -> Explanation:
    return Explanation(feature=feature, value=value, contribution=round(contribution, 3), text=text)

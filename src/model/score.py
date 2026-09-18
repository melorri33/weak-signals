"""Скоринг кандидатов: `score(candidates, features) -> list[ScoredCandidate]`.

ЗАГЛУШКА. Сигнатура финальная; вместо CatBoost + SHAP пока прозрачная эвристика:
каждое правило даёт вклад в логит (как SHAP-значение), уверенность = сигмоида суммы.
Вызов логируется как вызов модели (требование ТЗ).
"""

from __future__ import annotations

import math
from datetime import date

from src.common.logs import model_timer
from src.common.schemas import Candidate, CandidateFeatures, Explanation, ScoredCandidate

MODEL_NAME = "heuristic-stub-v0"
MODEL_PROVIDER = "local"
TOP_REASONS = 3


def score(candidates: list[Candidate], features: list[CandidateFeatures]) -> list[ScoredCandidate]:
    """Оценить кандидатов. Результат отсортирован по убыванию уверенности.

    Кандидат без признаков получает нейтральные 0.5 без причин.
    """
    by_id = {f.candidate_id: f for f in features}
    with model_timer(step="score", model=MODEL_NAME, provider=MODEL_PROVIDER):
        scored = [_score_one(c, by_id.get(c.id)) for c in candidates]
    return sorted(scored, key=lambda s: s.score, reverse=True)


def _score_one(candidate: Candidate, f: CandidateFeatures | None) -> ScoredCandidate:
    reasons = _reasons(f) if f else []
    logit = sum(r.contribution for r in reasons)
    meaningful = [r for r in reasons if r.contribution != 0]
    top = sorted(meaningful, key=lambda r: abs(r.contribution), reverse=True)[:TOP_REASONS]
    return ScoredCandidate(
        candidate_id=candidate.id,
        name=candidate.name,
        score=round(1 / (1 + math.exp(-logit)), 4),
        top_reasons=top,
    )


def _reasons(f: CandidateFeatures) -> list[Explanation]:
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

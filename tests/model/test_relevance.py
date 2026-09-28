"""Поправки ранжирования по эмбеддингам: близость к теме запроса и почти-дубли — на заглушке вместо bge-m3."""

import numpy as np
import pytest

from src.common.config import get_settings
from src.common.schemas import ScoredCandidate
from src.model import relevance

# Двумерные «эмбеддинги»: ось x — агротех, ось y — чужая тема.
VECTORS = {
    "перспективные технологии в агротехе": [1.0, 0.0],
    "soil moisture sensing": [1.0, 0.0],
    "weed spraying drone": [0.95, 0.31],
    "crop yield forecasting": [0.9, 0.44],
    "drug screening organoids": [0.0, 1.0],
    "weed spraying drones": [0.94, 0.33],
}


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    def embed(texts, step):
        v = np.array([VECTORS[t] for t in texts], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(relevance, "_embed", embed)


def _scored(*pairs: tuple[str, float]) -> list[ScoredCandidate]:
    return [ScoredCandidate(candidate_id=f"c{i}", name=n, score=s) for i, (n, s) in enumerate(pairs)]


def test_candidate_close_to_topic_overtakes_foreign_one():
    scored = _scored(("drug screening organoids", 0.80), ("weed spraying drone", 0.78), ("crop yield forecasting", 0.5))

    out = relevance.with_topic_relevance(scored, "перспективные технологии в агротехе", ["soil moisture sensing"])

    assert [s.name for s in out][:2] == ["weed spraying drone", "drug screening organoids"]
    reason = next(r for r in out[0].top_reasons if r.feature == "topic_relevance")
    assert reason.contribution > 0 and reason.text == "Близко к теме запроса"
    foreign = next(s for s in out if s.name == "drug screening organoids")
    assert not any(r.feature == "topic_relevance" for r in foreign.top_reasons)  # довод «против» не показываем


def test_fresh_phrases_do_not_define_the_topic_and_zero_weight_changes_nothing(monkeypatch: pytest.MonkeyPatch):
    scored = _scored(("drug screening organoids", 0.80), ("weed spraying drone", 0.78))
    phrases = ["soil moisture sensing", "agtech startup raises"]  # фраза свежести в тему не входит
    monkeypatch.setitem(VECTORS, "agtech startup raises", [0.0, 1.0])

    out = relevance.with_topic_relevance(scored, "перспективные технологии в агротехе", phrases)
    assert out[0].name == "weed spraying drone"

    monkeypatch.setattr(get_settings(), "topic_relevance_weight", 0.0)
    assert relevance.with_topic_relevance(scored, "перспективные технологии в агротехе", phrases) == scored


def test_near_duplicate_below_is_dropped_distinct_ones_stay():
    scored = _scored(("weed spraying drone", 0.9), ("drug screening organoids", 0.8), ("weed spraying drones", 0.7))

    assert [s.name for s in relevance.drop_near_duplicates(scored)] == [
        "weed spraying drone",
        "drug screening organoids",
    ]


def test_no_embeddings_keeps_the_list(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(relevance, "_embed", lambda texts, step: None)
    scored = _scored(("drug screening organoids", 0.8), ("weed spraying drone", 0.7))

    assert relevance.with_topic_relevance(scored, "перспективные технологии в агротехе", []) == scored
    assert relevance.drop_near_duplicates(scored) == scored

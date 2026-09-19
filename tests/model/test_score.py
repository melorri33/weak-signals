"""Офлайн-тесты model.score: эвристика (без файла модели) и обученная модель (если файл есть)."""

import importlib
from datetime import date

import pytest

from src.common.logs import collected_model_calls, start_run_log
from src.common.schemas import Candidate, CandidateFeatures, TermStats
from src.features import compute
from src.model import score

score_module = importlib.import_module("src.model.score")  # модуль, а не одноимённая функция из src.model

YEAR = date.today().year
LAST = YEAR - 1


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(score_module, "_load_model", lambda: None)


def test_heuristic_young_growing_beats_mature(no_model):
    cands = [
        Candidate(id="mature", name="Blockchain"),
        Candidate(id="young", name="ZK KYC"),
        Candidate(id="empty", name="?"),
    ]
    feats = [
        CandidateFeatures(
            candidate_id="mature", total_pubs=45000, first_seen_year=2008, has_standard=True, has_wikipedia=True
        ),
        CandidateFeatures(
            candidate_id="young", total_pubs=150, growth_3y=0.8, first_seen_year=YEAR - 4, news_to_science_ratio=0.2
        ),
    ]

    result = score(cands, feats)

    assert [s.candidate_id for s in result] == ["young", "empty", "mature"]
    young, empty, mature = result
    assert young.score > 0.75
    assert mature.score < 0.25
    assert empty.score == 0.5 and empty.top_reasons == []
    assert all(r.contribution > 0 for r in young.top_reasons)


def test_score_is_logged_as_model_call(no_model):
    start_run_log()
    score([Candidate(id="a", name="A")], [])
    assert [c.step for c in collected_model_calls()] == ["score"]


@pytest.mark.skipif(not score_module.MODEL_PATH.exists(), reason="модель не обучена")
def test_trained_model_prefers_small_growing_topics():
    """Суть модели: при равном объёме растущая тема выше плоской; маленькая растущая — выше зрелой."""
    no_news = {2016: 0, 2023: 0, 2024: 1, LAST: 1}
    growing = TermStats(term="g", pubs_by_year={LAST - 3: 2, LAST - 2: 5, LAST - 1: 12, LAST: 30}, news_by_year=no_news)
    flat = TermStats(term="f", pubs_by_year={y: 30 for y in range(LAST - 8, LAST + 1)}, news_by_year=no_news)
    mature = TermStats(
        term="m",
        pubs_by_year={y: 3000 + 200 * (y - 2000) for y in range(2000, YEAR + 1)},
        news_by_year={2016: 2000, 2023: 400, 2024: 400, LAST: 400},
        wikipedia_en=True,
    )
    cands = [Candidate(id=k, name=k) for k in ("growing", "flat", "mature")]
    feats = [compute(c, [], s) for c, s in zip(cands, (growing, flat, mature), strict=True)]

    start_run_log()
    result = {s.candidate_id: s for s in score(cands, feats)}

    assert result["growing"].score > result["flat"].score > result["mature"].score
    assert result["mature"].score < 0.2
    assert result["growing"].top_reasons and all(r.text for r in result["growing"].top_reasons)
    assert collected_model_calls()[-1].model == score_module.MODEL_NAME

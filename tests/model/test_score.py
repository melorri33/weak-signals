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
def test_trained_model_separates_young_from_mature():
    young = TermStats(
        term="young",
        pubs_by_year={LAST - 3: 5, LAST - 2: 15, LAST - 1: 40, LAST: 90},
        news_by_year={LAST: 1},
        wikipedia_en=False,
    )
    mature = TermStats(
        term="mature",
        pubs_by_year={y: 3000 + 200 * (y - 2000) for y in range(2000, YEAR + 1)},
        news_by_year={y: 300 for y in range(2016, YEAR + 1)},
        wikipedia_en=True,
    )
    cands = [Candidate(id="young", name="young"), Candidate(id="mature", name="mature")]
    feats = [compute(cands[0], [], young), compute(cands[1], [], mature)]

    start_run_log()
    result = {s.candidate_id: s for s in score(cands, feats)}

    assert result["young"].score > 0.5 > result["mature"].score
    assert result["young"].top_reasons and all(r.text for r in result["young"].top_reasons)
    assert collected_model_calls()[-1].model == score_module.MODEL_NAME

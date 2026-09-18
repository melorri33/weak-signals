"""Офлайн-тесты заглушки model.score."""

from datetime import date

from src.common.logs import collected_model_calls, start_run_log
from src.common.schemas import Candidate, CandidateFeatures
from src.model import score

YEAR = date.today().year


def test_young_growing_beats_mature():
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
    assert all(0 <= s.score <= 1 for s in result)
    young, empty, mature = result
    assert young.score > 0.75
    assert mature.score < 0.25
    assert empty.score == 0.5 and empty.top_reasons == []
    assert 0 < len(young.top_reasons) <= 3
    assert all(r.contribution > 0 for r in young.top_reasons)
    assert all(r.text for r in mature.top_reasons)


def test_score_is_logged_as_model_call():
    start_run_log()
    score([Candidate(id="a", name="A")], [])
    calls = collected_model_calls()
    assert [c.step for c in calls] == ["score"]

"""Офлайн-тесты заглушки features.compute."""

from datetime import date

from src.common.schemas import Candidate, Document, SourceType, TermStats
from src.features import compute

LAST = date.today().year - 1


def _doc(doc_id: str, source: str, source_type: SourceType) -> Document:
    return Document(id=doc_id, source=source, source_type=source_type, title="t", url=f"https://example.org/{doc_id}")


def test_features_from_stats_and_docs():
    cand = Candidate(id="zk-kyc", name="Zero-knowledge KYC", document_ids=["a", "b", "c"])
    docs = [
        _doc("a", "openalex", SourceType.PAPER),
        _doc("b", "arxiv", SourceType.PREPRINT),
        _doc("c", "gdelt", SourceType.NEWS),
        _doc("x", "openalex", SourceType.PAPER),  # чужой документ — не учитывается
    ]
    stats = TermStats(
        term="zero-knowledge kyc",
        pubs_by_year={LAST - 3: 10, LAST - 2: 20, LAST - 1: 40, LAST: 80},
        news_by_year={LAST: 15},
        wikipedia_ru=False,
        wikipedia_en=False,
        standard_mentions=0,
    )

    f = compute(cand, docs, stats)

    assert f.candidate_id == "zk-kyc"
    assert f.total_pubs == 150
    assert f.growth_3y == 1.0  # 10 → 80 за 3 года = удвоение каждый год
    assert f.first_seen_year == LAST - 3
    assert f.news_total == 15
    assert f.news_to_science_ratio == 0.1
    assert f.distinct_sources == 3
    assert f.has_wikipedia is False
    assert f.has_standard is False
    assert f.patents_total is None  # патенты не пришли — None, а не 0


def test_no_stats_gives_none_not_zero():
    cand = Candidate(id="c1", name="Something", document_ids=["a"])
    f = compute(cand, [_doc("a", "openalex", SourceType.PAPER)], None)

    assert f.total_pubs is None
    assert f.growth_3y is None
    assert f.has_wikipedia is None
    assert f.news_to_science_ratio == 0.0  # посчитано по документам: 0 новостей / 1 статья
    assert f.distinct_sources == 1


def test_preprint_share_from_types():
    cand = Candidate(id="c", name="c")
    stats = TermStats(term="c", pubs_by_year={LAST: 10}, pubs_by_type={"preprint": 3, "article": 6, "review": 1})
    assert compute(cand, [], stats).extra["preprint_share"] == 0.3
    assert compute(cand, [], TermStats(term="c", pubs_by_year={LAST: 10})).extra["preprint_share"] is None

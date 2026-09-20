"""Правила отсева: срабатывают только на очевидном и объясняют причину."""

from pathlib import Path

import pytest

from src.common.schemas import Candidate, CandidateFeatures, Document, SourceType, TrustLevel
from src.features import compute
from src.filters import apply

CAND = Candidate(id="c", name="c")


def _f(**kw) -> CandidateFeatures:
    return CandidateFeatures(candidate_id="c", **kw)


def test_mature_by_publications_with_reason():
    d = apply(CAND, _f(total_pubs=45000))
    assert (d.excluded, d.reason_code) == (True, "mature")
    assert "45 000 публикаций" in d.reason_text


def test_mature_by_standard():
    d = apply(CAND, _f(total_pubs=100, has_standard=True))
    assert d.reason_code == "mature" and "стандарт" in d.reason_text


def test_hype_needs_both_media_volume_and_ratio():
    assert apply(CAND, _f(total_pubs=300, news_total=400, news_to_science_ratio=1.3)).reason_code == "hype"
    assert not apply(CAND, _f(total_pubs=300, news_total=66, news_to_science_ratio=1.15)).excluded  # как у сигналов
    assert not apply(CAND, _f(total_pubs=3000, news_total=400, news_to_science_ratio=0.13)).excluded


def test_missing_data_never_excludes():
    d = apply(CAND, _f())
    assert (d.excluded, d.reason_code) == (False, "ok")


def _doc(i: str, trust: TrustLevel | None, st: SourceType = SourceType.BLOG) -> Document:
    return Document(id=i, source="rss", source_type=st, title="t", url=f"https://x/{i}", trust=trust)


def test_noise_when_only_low_trust_sources():
    cand = Candidate(id="c", name="c", document_ids=["a", "b"])
    low_only = compute(cand, [_doc("a", TrustLevel.LOW), _doc("b", TrustLevel.LOW)], None)
    d = apply(cand, low_only)
    assert d.reason_code == "noise" and "2 источника" in d.reason_text

    confirmed = compute(cand, [_doc("a", TrustLevel.LOW), _doc("b", TrustLevel.MEDIUM, SourceType.NEWS)], None)
    assert not apply(cand, confirmed).excluded

    unrated = compute(cand, [_doc("a", None), _doc("b", None)], None)
    assert not apply(cand, unrated).excluded


@pytest.mark.skipif(not Path("data/labeled_set.csv").exists(), reason="нужны данные организаторов (есть у artiom)")
def test_rules_never_exclude_organizers_signals():
    """Главное требование к правилам: ни один сигнал из датасета организаторов не отсекается."""
    import pandas as pd

    from src.model import train

    labeled = pd.read_csv("data/labeled_set.csv")
    stats = train.load_term_stats()
    excluded = [
        r.term_en
        for r in labeled[labeled.label == 1].itertuples()
        if apply(
            Candidate(id=r.key, name=r.term_en), compute(Candidate(id=r.key, name=r.term_en), [], stats[r.term_en])
        ).excluded
    ]
    assert excluded == []


def test_product_names_without_research_are_excluded():
    """Из новостей в кандидаты попадают названия продуктов («Gemini 3.8 Flash»): по точной фразе
    у них нет научных работ, и в выдачу они попадать не должны."""
    decision = apply(
        Candidate(id="p", name="какой-то продукт"),
        CandidateFeatures(candidate_id="p", total_pubs=0, extra={"docs_rated": 1.0, "docs_trusted": 1.0}),
    )

    assert decision.excluded
    assert decision.reason_code == "no_research"
    assert "продукта" in decision.reason_text


def test_missing_publication_count_does_not_exclude():
    """Источник не ответил — это не повод исключать: пустой признак не считается плохим."""
    decision = apply(
        Candidate(id="p", name="термин"),
        CandidateFeatures(candidate_id="p", total_pubs=None, extra={"docs_rated": 1.0, "docs_trusted": 1.0}),
    )

    assert not decision.excluded

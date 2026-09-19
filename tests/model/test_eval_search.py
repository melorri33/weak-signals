"""Офлайн-тесты проверки «как у жюри» на выдуманном эталоне (файл организаторов не нужен)."""

from src.common.schemas import Candidate, Document, SearchResult, SignalCard, SourceRef, SourceType, TrustLevel
from src.model.eval_search import ReferenceItem, _split_companies, evaluate, match, report_md

REF = [
    ReferenceItem(1, "Квантовые сенсоры", "Защита ИИ", "quantum sensing", ["Quantum Diamonds", "qsense"]),
    ReferenceItem(2, "Твердотельные аккумуляторы", "Защита ИИ", "solid-state battery", ["Batterix"]),
    ReferenceItem(3, "Натрий-ионные накопители", "Финтех", "sodium-ion storage", ["Natrion Grid"]),
]


def _card(name: str, source_title: str) -> SignalCard:
    src = SourceRef(
        document_id="d",
        title=source_title,
        url="https://example.org",
        source_type=SourceType.NEWS,
        language="en",
        trust=TrustLevel.MEDIUM,
    )
    return SignalCard(
        candidate_id=name,
        name=name,
        score=0.9,
        description="",
        advantage="",
        case_example="",
        why_weak_signal="",
        sources=[src],
    )


def test_split_companies_drops_russian_notes():
    raw = "Multiverse Computing (CompactifAI), Scania Invest (стратеги-инвесторы), академические группы"
    assert _split_companies(raw) == ["Multiverse Computing", "CompactifAI", "Scania Invest"]


def test_match_by_company_and_term_with_word_boundaries():
    hits = {i.id: by for i, by in match("Batterix raises $30M; new quantum sensing scanner", REF)}
    assert hits == {2: "Batterix", 1: "термин «quantum sensing»"}
    assert match("qsensor released", REF) == []  # «qsense» не должен совпасть с «qsensor»


def test_evaluate_shows_where_items_are_lost():
    docs = [Document(id="a", source="rss", source_type=SourceType.NEWS, title="Quantum Diamonds and Batterix", url="u")]
    cands = [Candidate(id="c1", name="quantum sensing")]
    result = SearchResult(run_id="r", query="q", top=[_card("quantum sensors", "Quantum Diamonds launches qsense")])

    stages = evaluate(REF, "Защита ИИ", result=result, documents=docs, candidates=cands)

    assert [(s.stage, sorted(s.found)) for s in stages] == [
        ("документы после сбора", [1, 2]),
        ("кандидаты", [1]),
        ("топ-15", [1]),
    ]
    md = report_md(REF, "Защита ИИ", "q", stages)
    assert "✅ Квантовые сенсоры" in md and "❌ Твердотельные аккумуляторы" in md


def test_scored_stage_separates_found_but_ranked_low():
    from src.common.schemas import ScoredCandidate

    scored = [
        ScoredCandidate(candidate_id=f"c{i}", name=n, score=0.9 - i / 100)
        for i, n in enumerate(["x", "Batterix platform"])
    ]
    result = SearchResult(run_id="r", query="q", top=[], scored=scored)
    stages = {s.stage: sorted(s.found) for s in evaluate(REF, "Защита ИИ", result=result)}
    assert stages == {"все проскоренные": [2], "топ-15": []}


def test_summary_counts_per_stage_and_total():
    from src.model.eval_search import StageReport, summary_md

    per_domain = {
        "Защита ИИ": [StageReport("кандидаты", {1: "x", 2: "y"}), StageReport("топ-15", {1: "x"})],
        "Финтех": [StageReport("кандидаты", {}), StageReport("топ-15", {})],
    }
    md = summary_md(REF, per_domain)
    assert "| Защита ИИ | 2 | 2 | 1 |" in md
    assert "| **Всего** | 3 | **2** | **1** |" in md


def test_queries_yaml_covers_all_dataset_domains_and_flags_mature():
    from src.model.eval_search import DOMAIN_QUERIES, TEST_QUERIES, must_exclude_violations

    assert set(DOMAIN_QUERIES) == {"Индустриальный ИИ", "Инфраструктура ИИ", "Роботы", "Финтех", "Edge", "Защита ИИ"}
    assert all(q["query"] and q["must_exclude"] and q["expected_like"] for q in TEST_QUERIES)
    result = SearchResult(
        run_id="r", query=DOMAIN_QUERIES["Финтех"], top=[_card("Mobile banking", "x"), _card("sodium-ion storage", "y")]
    )
    assert must_exclude_violations(result) == ["mobile banking"]

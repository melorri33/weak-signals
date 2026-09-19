"""Офлайн-тесты проверки «как у жюри» на выдуманном эталоне (файл организаторов не нужен)."""

from src.common.schemas import Candidate, Document, SearchResult, SignalCard, SourceRef, SourceType, TrustLevel
from src.model.eval_search import ReferenceItem, _split_companies, evaluate, match, report_md

REF = [
    ReferenceItem(1, "Защита MCP-серверов", "Защита ИИ", "MCP security", ["Invariant Labs", "mcp-scan"]),
    ReferenceItem(2, "Red teaming как услуга", "Защита ИИ", "automated AI red teaming", ["Mindgard"]),
    ReferenceItem(3, "Токенизированные депозиты", "Финтех", "tokenized deposits", ["Kinexys"]),
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
    hits = {i.id: by for i, by in match("Mindgard raises $30M; new MCP security scanner", REF)}
    assert hits == {2: "Mindgard", 1: "термин «MCP security»"}
    assert match("mcp-scanner released", REF) == []  # «mcp-scan» не должен совпасть с «mcp-scanner»


def test_evaluate_shows_where_items_are_lost():
    docs = [Document(id="a", source="rss", source_type=SourceType.NEWS, title="Invariant Labs and Mindgard", url="u")]
    cands = [Candidate(id="c1", name="MCP security")]
    result = SearchResult(run_id="r", query="q", top=[_card("MCP server scanners", "Invariant Labs launches mcp-scan")])

    stages = evaluate(REF, "Защита ИИ", result=result, documents=docs, candidates=cands)

    assert [(s.stage, sorted(s.found)) for s in stages] == [
        ("документы после сбора", [1, 2]),
        ("кандидаты", [1]),
        ("топ-15", [1]),
    ]
    md = report_md(REF, "Защита ИИ", "q", stages)
    assert "✅ Защита MCP-серверов" in md and "❌ Red teaming как услуга" in md

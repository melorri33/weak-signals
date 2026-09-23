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
    card = _card("diamond quantum sensing", "Quantum Diamonds launches qsense")
    result = SearchResult(run_id="r", query="q", top=[card])

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


def test_top15_does_not_count_someone_elses_headline():
    """Термин в заголовке статьи из списка ссылок — не наша заслуга.

    Замер 21.09 по здоровому прогону: все до единого совпадения старой проверки приходили
    именно оттуда. Жюри сверяет название и суть сигнала, а не то, на что мы сослались.
    """
    card = _card("edge inference accelerator", "Quantum sensing goes commercial")
    result = SearchResult(run_id="r", query="q", top=[card])

    stages = evaluate(REF, "Защита ИИ", result=result)

    assert stages[-1].stage == "топ-15"
    assert stages[-1].found == {}


def test_top15_counts_our_name_across_word_forms():
    """«quantum sensors» и «quantum sensing» — одна технология, множественное число не помеха."""
    result = SearchResult(run_id="r", query="q", top=[_card("photonic quantum sensing", "нейтральный заголовок")])

    stages = evaluate(REF, "Защита ИИ", result=result)

    assert sorted(stages[-1].found) == [1]
    assert "название" in stages[-1].found[1]


def test_top15_rejects_a_name_broader_than_the_dataset_one():
    """Наше название может быть уже датасетного, но не шире: широкое эксперт не засчитает."""
    result = SearchResult(run_id="r", query="q", top=[_card("battery", "нейтральный заголовок")])

    stages = evaluate(REF, "Защита ИИ", result=result)

    assert stages[-1].found == {}


def test_top15_counts_the_technology_named_in_the_description():
    """Назвали в описании карточки — засчитываем: организаторы сверяют и суть тоже."""
    card = _card("новая химия накопителей", "нейтральный заголовок")
    card.description = "Разработчики применяют solid-state battery в серийных модулях."
    result = SearchResult(run_id="r", query="q", top=[card])

    stages = evaluate(REF, "Защита ИИ", result=result)

    assert sorted(stages[-1].found) == [2]
    assert "описание" in stages[-1].found[2]

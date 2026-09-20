"""Проверка открытого поиска «как у жюри»: запрос по области датасета → топ-15 → сколько совпало с датасетом.

Организаторы проверяют именно так (ответ 19.09): по технологиям датасета делают открытый запрос и считают,
сколько из нашего топ-15 совпало с их списком. Здесь то же самое автоматически, плюс диагностика —
на каком шаге теряются технологии: документы после сбора → кандидаты → все проскоренные → топ-15.

Сопоставление — по компаниям из датасета и по термину технологии (data/positive_terms.csv).
Они используются ТОЛЬКО для проверки: вшивать их в поиск или промпты запрещает ТЗ.
Сопоставление автоматическое и приблизительное — спорные случаи смотреть глазами (отчёт перечисляет, что с чем совпало).

Запуск: python -m src.model.eval_search result.json [--domain Финтех]
        python -m src.model.eval_search --run-all   # все 6 областей → data/search_eval.md
        python -m src.model.eval_search --ceiling   # потолок: технологии датасета в собранных документах
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.common.logs import get_logger
from src.common.schemas import Candidate, Document, SearchResult
from src.model.dataset import load_signals

DATA = Path("data")
TERMS_PATH = DATA / "positive_terms.csv"
QUERIES_PATH = Path(__file__).resolve().parents[2] / "tests" / "queries.yaml"

log = get_logger(__name__)

# Тестовые запросы (tests/queries.yaml); у запросов по областям датасета есть domain.
TEST_QUERIES: list[dict] = yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
# Запрос жюри по каждой области датасета — наш вариант формулировки (точных формулировок жюри мы не знаем).
DOMAIN_QUERIES: dict[str, str] = {q["domain"]: q["query"] for q in TEST_QUERIES if q.get("domain")}
MUST_EXCLUDE: dict[str, list[str]] = {q["query"]: q.get("must_exclude", []) for q in TEST_QUERIES}


def must_exclude_violations(result: SearchResult) -> list[str]:
    """Очевидно зрелые технологии из tests/queries.yaml, которые всё-таки попали в топ-15."""
    names = [f"{c.name}\n{c.name_ru or ''}" for c in result.top]
    return [m for m in MUST_EXCLUDE.get(result.query, []) if any(_mentions(n, m) for n in names)]


# Крупные компании встречаются в новостях о чём угодно — по ним совпадение ничего не доказывает.
_BIG_COMPANIES = {
    "google",
    "microsoft",
    "nvidia",
    "amazon",
    "meta",
    "apple",
    "samsung",
    "intel",
    "amd",
    "cisco",
    "ibm",
    "qualcomm",
    "hp",
    "palo alto",
    "siemens",
    "visa",
    "stripe",
    "coinbase",
    "j.p. morgan",
    "jpmorgan",
    "openai",
    "anthropic",
    "claude",
    "check point",
    "dell",
    "arm",
    "sony",
    "huawei",
    "bosch",
    "honda",
    "akamai",
    "cloudflare",
    "swift",
    "google cloud",
}
_MIN_NAME_LEN = 4


def _is_big(name: str) -> bool:
    """«Siemens Technology», «HP Inc», «Microsoft Security Research» — тоже крупные компании."""
    low = name.lower()
    return any(low == big or low.startswith(big + " ") for big in _BIG_COMPANIES)


@dataclass
class ReferenceItem:
    id: int
    name_ru: str
    domain: str
    term_en: str
    companies: list[str] = field(default_factory=list)


def _split_companies(raw: str) -> list[str]:
    """«Multiverse Computing (CompactifAI), HP Inc.» → [Multiverse Computing, CompactifAI, HP Inc.].

    Русские пояснения в скобках отбрасываются.
    """
    names = []
    for part in re.split(r"[,;/+]|\s[—–-]\s", raw):
        for name in re.split(r"[()]", part):
            name = name.strip(" .«»\"'")
            if len(name) >= _MIN_NAME_LEN and not re.search(r"[а-яё]", name, re.IGNORECASE):
                names.append(name)
    return names


def load_reference(xlsx_path: Path | None = None, terms_path: Path = TERMS_PATH) -> list[ReferenceItem]:
    """Технологии датасета с компаниями; компании, встречающиеся у нескольких технологий, убираются."""
    xlsx_path = xlsx_path or Path(glob.glob(str(DATA / "*.xlsx"))[0])
    df = load_signals(xlsx_path, terms_path)
    per_item = {int(r.id): _split_companies(str(r.companies)) for r in df.itertuples()}
    freq = Counter(n.lower() for names in per_item.values() for n in set(names))
    return [
        ReferenceItem(
            id=int(r.id),
            name_ru=r.name_ru,
            domain=r.domain,
            term_en=r.term_en,
            companies=[n for n in per_item[int(r.id)] if freq[n.lower()] == 1 and not _is_big(n)],
        )
        for r in df.itertuples()
    ]


def _mentions(text: str, needle: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(needle)}(?![\w-])", text, re.IGNORECASE) is not None


def match(text: str, reference: list[ReferenceItem]) -> list[tuple[ReferenceItem, str]]:
    """Какие технологии датасета упомянуты в тексте и по чему: компания или термин."""
    hits = []
    for item in reference:
        by = next((c for c in item.companies if _mentions(text, c)), None)
        if by is None and _mentions(text, item.term_en):
            by = f"термин «{item.term_en}»"
        if by is not None:
            hits.append((item, by))
    return hits


def _card_text(card) -> str:
    parts = [card.name, card.name_ru or "", card.description, card.case_example]
    parts += [s.title for s in card.sources] + [s.ru_summary or "" for s in card.sources]
    return "\n".join(parts)


@dataclass
class StageReport:
    stage: str
    found: dict[int, str]  # id технологии датасета → чем подтверждено


def evaluate(
    reference: list[ReferenceItem],
    domain: str,
    result: SearchResult | None = None,
    documents: list[Document] | None = None,
    candidates: list[Candidate] | None = None,
) -> list[StageReport]:
    """Сколько технологий области нашлось на каждом доступном шаге конвейера."""
    items = [i for i in reference if i.domain == domain]
    stages: list[StageReport] = []
    if documents is not None:
        text = "\n".join(f"{d.title}\n{d.abstract or ''}" for d in documents)
        stages.append(StageReport("документы после сбора", {i.id: by for i, by in match(text, items)}))
    if candidates is not None:
        text = "\n".join("\n".join([c.name, c.name_ru or "", *c.aliases]) for c in candidates)
        stages.append(StageReport("кандидаты", {i.id: by for i, by in match(text, items)}))
    if result is not None and result.scored:
        text = "\n".join(s.name for s in result.scored)
        stages.append(StageReport("все проскоренные", {i.id: by for i, by in match(text, items)}))
    if result is not None:
        found: dict[int, str] = {}
        for card in result.top:
            for item, by in match(_card_text(card), items):
                found.setdefault(item.id, f"{by} → «{card.name_ru or card.name}»")
        stages.append(StageReport("топ-15", found))
    return stages


def report_md(reference: list[ReferenceItem], domain: str, query: str, stages: list[StageReport]) -> str:
    items = [i for i in reference if i.domain == domain]
    lines = [f"## {domain}: «{query}»", "", f"Технологий области в датасете: {len(items)}", ""]
    lines += ["| Шаг | Найдено | Доля |", "| --- | --- | --- |"]
    lines += [f"| {s.stage} | {len(s.found)} | {len(s.found) / len(items):.0%} |" for s in stages]
    if stages:
        last = stages[-1]
        lines += ["", f"Совпадения на шаге «{last.stage}»:", ""]
        lines += [f"- ✅ {i.name_ru} — {last.found[i.id]}" for i in items if i.id in last.found]
        lines += [f"- ❌ {i.name_ru}" for i in items if i.id not in last.found]
    return "\n".join(lines) + "\n"


def _domain_from_query(query: str) -> str | None:
    return next((d for d, q in DOMAIN_QUERIES.items() if q == query), None)


def summary_md(reference: list[ReferenceItem], per_domain: dict[str, list[StageReport]]) -> str:
    """Сводка по всем областям: сколько технологий датасета дошло до каждого шага."""
    stage_names = list(dict.fromkeys(s.stage for stages in per_domain.values() for s in stages))
    lines = ["| Область | В датасете | " + " | ".join(stage_names) + " |", "| --- " * (len(stage_names) + 2) + "|"]
    totals = dict.fromkeys(stage_names, 0)
    n_total = 0
    for domain, stages in per_domain.items():
        n = sum(1 for i in reference if i.domain == domain)
        n_total += n
        found = {s.stage: len(s.found) for s in stages}
        for name in stage_names:
            totals[name] += found.get(name, 0)
        lines.append(f"| {domain} | {n} | " + " | ".join(str(found.get(name, "—")) for name in stage_names) + " |")
    lines.append(f"| **Всего** | {n_total} | " + " | ".join(f"**{totals[n]}**" for n in stage_names) + " |")
    return "\n".join(lines) + "\n"


async def run_all(out_dir: Path) -> str:
    """Прогнать конвейер по всем областям датасета, сохранить результаты и собрать отчёт."""
    from src.pipeline.run import run  # импорт здесь: оценка одного JSON не должна тянуть конвейер

    reference = load_reference()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_domain: dict[str, list[StageReport]] = {}
    sections = []
    for domain, query in DOMAIN_QUERIES.items():
        result = await run(query)
        (out_dir / f"{domain}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        per_domain[domain] = evaluate(reference, domain, result=result)
        section = report_md(reference, domain, query, per_domain[domain])
        if violations := must_exclude_violations(result):
            section += f"\n⚠️ В топ-15 попало очевидно зрелое (must_exclude): {', '.join(violations)}\n"
        sections.append(section)
    return "\n".join(["# Проверка поиска «как у жюри»", "", summary_md(reference, per_domain), *sections])


async def ceiling(out_dir: Path) -> str:
    """Потолок выдачи: сколько технологий датасета вообще попадает в собранные документы.

    Это главное число проекта (docs/HANDOFF.md): чего нет в документах, того не будет ни
    в кандидатах, ни в топ-15, сколько модель ни улучшай. Считается только первыми двумя
    шагами конвейера — расширение запроса и сбор, — поэтому шесть областей проходят
    за минуты, а не за полчаса.

    Полный --run-all этого числа не даёт: SearchResult документов не содержит, и шаг
    «документы после сбора» в нём всегда пропускается.
    """
    from src.common.config import get_settings
    from src.llm.expand_query import expand_query
    from src.pipeline import deps

    settings = get_settings()
    reference = load_reference()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, int, int, int]] = []
    for domain, query in DOMAIN_QUERIES.items():
        phrases = await expand_query(query) or [query]
        docs = await deps.collect(phrases, limit=settings.max_documents)
        stages = evaluate(reference, domain, documents=docs)
        found = stages[0].found if stages else {}
        total = sum(1 for i in reference if i.domain == domain)
        rows.append((domain, len(phrases), len(docs), len(found), total))
        log.info("Потолок %s: документов %d, технологий области %d из %d", domain, len(docs), len(found), total)
        (out_dir / f"{domain}_docs.json").write_text(_dumps_titles(docs), encoding="utf-8")
    return _ceiling_md(rows)


def _dumps_titles(docs: list[Document]) -> str:
    """Заголовки собранных документов — чтобы потолок можно было перепроверить глазами."""
    rows = [{"title": d.title, "url": d.url, "source": d.source, "type": d.source_type.value} for d in docs]
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _ceiling_md(rows: list[tuple[str, int, int, int, int]]) -> str:
    lines = [
        "# Потолок выдачи: технологии датасета в собранных документах",
        "",
        "Чего нет здесь — не появится ни в кандидатах, ни в топ-15.",
        "",
        "| Область | Фраз | Документов | Технологий найдено | Всего в области |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [f"| {d} | {p} | {n} | {f} | {t} |" for d, p, n, f, t in rows]
    docs_total = sum(r[2] for r in rows)
    found_total = sum(r[3] for r in rows)
    all_total = sum(r[4] for r in rows)
    lines.append(f"| **Всего** | | **{docs_total}** | **{found_total}** | **{all_total}** |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Сколько технологий датасета попало в топ-15")
    ap.add_argument("result", type=Path, nargs="?", help="SearchResult в JSON")
    ap.add_argument("--domain", choices=list(DOMAIN_QUERIES), help="область датасета (иначе — по тексту запроса)")
    ap.add_argument("--run-all", action="store_true", help="прогнать конвейер по всем 6 областям и собрать отчёт")
    ap.add_argument("--ceiling", action="store_true", help="только сбор: технологии датасета в документах")
    args = ap.parse_args()
    if args.ceiling:
        import asyncio

        report = asyncio.run(ceiling(DATA / "ceiling"))
        path = DATA / "search_ceiling.md"
        path.write_text(report, encoding="utf-8")
        print(report)
        print(f"Отчёт: {path}; заголовки документов: {DATA / 'ceiling'}")
        return
    if args.run_all:
        import asyncio

        report = asyncio.run(run_all(DATA / "search_runs"))
        # В data/, а не в reports/: отчёт перечисляет все технологии датасета, а репозиторий публичный.
        path = DATA / "search_eval.md"
        path.write_text(report, encoding="utf-8")
        print(report)
        print(f"Отчёт: {path}; результаты прогонов: {DATA / 'search_runs'}")
        return
    if args.result is None:
        ap.error("укажи файл результата или --run-all")
    result = SearchResult.model_validate_json(args.result.read_text(encoding="utf-8"))
    domain = args.domain or _domain_from_query(result.query)
    if domain is None:
        raise SystemExit(f"Не понял область по запросу «{result.query}» — укажи --domain")
    reference = load_reference()
    print(report_md(reference, domain, result.query, evaluate(reference, domain, result=result)))


if __name__ == "__main__":
    main()

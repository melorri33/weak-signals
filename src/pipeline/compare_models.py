"""Сравнение языковых моделей на запросах жюри: один и тот же конвейер, разные модели, одна таблица.

Та же проверка, что `python -m src.model.eval_search --run-all` (6 областей датасета из tests/queries.yaml),
но по нескольким моделям подряд и с возобновлением — чтобы оставить на ночь:

    python -m src.pipeline.compare_models gigachat:GigaChat-2 yandexgpt:yandexgpt-5-lite ollama:qwen3:4b
    python -m src.pipeline.compare_models ollama:qwen3:4b --domains Финтех   # короткая проба
    python -m src.pipeline.compare_models --report-only                      # пересобрать отчёт

- Результат каждой области — data/model_runs/<провайдер>__<модель>/<область>.json. Готовая (успешная)
  область при повторном запуске пропускается: упал ночью — запусти ту же команду, доделается только недостающее.
- Каждая модель идёт отдельным процессом: настройки (get_settings) кэшируются на процесс, и модели
  иначе смешались бы.
- Нет ключа к облаку — модель пропускается с записью в лог, остальные идут дальше.
- Отчёт — data/model_compare.md, не в репозитории: в нём названия технологий датасета.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.common.logs import get_logger
from src.common.schemas import SearchResult

log = get_logger(__name__)

DATA = Path("data")
RUNS_DIR = DATA / "model_runs"
REPORT_PATH = DATA / "model_compare.md"
# Описание, которое ставит make_card, если модель не ответила (src/llm/cards.py).
NO_TEXT_PREFIX = "Описание не сформировано"
# Шаги журнала моделей, где работает не языковая модель: классификатор и модель названий.
NOT_LLM_STEPS = {"score", "name_model"}


def parse_spec(spec: str) -> tuple[str, str]:
    """«ollama:qwen3:4b» → («ollama», «qwen3:4b»): провайдер — до первого двоеточия."""
    provider, sep, model = spec.partition(":")
    if not sep or not provider or not model:
        raise ValueError(f"Модель задаётся как провайдер:модель, например ollama:qwen3:4b — получено «{spec}»")
    return provider, model


def run_dir(provider: str, model: str, root: Path = RUNS_DIR) -> Path:
    safe = re.sub(r"[^\w.-]", "-", model)
    return root / f"{provider}__{safe}"


def domain_queries() -> dict[str, str]:
    """Запросы по областям датасета — те же, что гоняет проверка «как у жюри»."""
    from src.model.eval_search import DOMAIN_QUERIES

    return DOMAIN_QUERIES


async def run_model(provider: str, model: str, domains: dict[str, str], root: Path = RUNS_DIR) -> int:
    """Прогнать одну модель по областям. Возвращает число новых прогонов. Модель берётся из окружения."""
    from src.llm.errors import LLMError
    from src.llm.providers import make_backend
    from src.pipeline.run import run

    try:
        make_backend(provider, model)
    except LLMError as exc:
        log.error("Модель %s:%s пропущена: %s", provider, model, exc)
        return 0
    out = run_dir(provider, model, root)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for domain, query in domains.items():
        path = out / f"{domain}.json"
        if _finished(path):
            log.info("%s:%s — «%s» уже есть, пропускаю", provider, model, domain)
            continue
        log.info("%s:%s — начинаю «%s»: %s", provider, model, domain, query)
        started = time.perf_counter()
        try:
            result = await run(query)
        except Exception:
            log.exception("%s:%s — «%s» упал, иду дальше", provider, model, domain)
            continue
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        done += 1
        log.info(
            "%s:%s — «%s» готово за %.0f с: в топе %d, уверенных %d",
            provider,
            model,
            domain,
            time.perf_counter() - started,
            len(result.top),
            result.confident_signals,
        )
    return done


def _finished(path: Path) -> bool:
    """Готовая область — только успешный прогон: упавший с ошибкой пересчитываем при следующем запуске."""
    if not path.exists():
        return False
    try:
        return SearchResult.model_validate_json(path.read_text(encoding="utf-8")).status == "done"
    except ValueError:
        return False


@dataclass
class Row:
    model: str
    domain: str
    duration_s: float
    documents: int
    candidates: int
    top: int
    confident: int
    no_text: int
    llm_calls: int
    llm_s: float
    failures: int
    matched_top: int | None = None
    matched_scored: int | None = None


def row_for(model: str, domain: str, result: SearchResult, reference=None) -> Row:
    llm = [c for c in result.model_calls if c.step not in NOT_LLM_STEPS]
    row = Row(
        model=model,
        domain=domain,
        duration_s=result.duration_s or 0.0,
        documents=result.documents_processed,
        candidates=result.candidates_found,
        top=len(result.top),
        confident=result.confident_signals,
        no_text=sum(1 for c in result.top if c.description.startswith(NO_TEXT_PREFIX)),
        llm_calls=len(llm),
        llm_s=sum(c.duration_ms for c in llm) / 1000,
        failures=len(result.source_failures),
    )
    if reference is not None:
        from src.model.eval_search import evaluate

        found = {s.stage: len(s.found) for s in evaluate(reference, domain, result=result)}
        row.matched_top = found.get("топ-15")
        row.matched_scored = found.get("все проскоренные")
    return row


def load_rows(root: Path = RUNS_DIR, reference=None) -> list[Row]:
    rows = []
    for path in sorted(root.glob("*__*/*.json")):
        model = path.parent.name.replace("__", ":", 1)
        result = SearchResult.model_validate_json(path.read_text(encoding="utf-8"))
        rows.append(row_for(model, path.stem, result, reference))
    return rows


def load_reference_if_present():
    """Технологии датасета для столбца «совпало»: только если файлы датасета лежат в data/."""
    if not glob.glob(str(DATA / "*.xlsx")) or not (DATA / "positive_terms.csv").exists():
        return None
    from src.model.eval_search import load_reference

    return load_reference()


def _n(value: int | None) -> str:
    return "—" if value is None else str(value)


def report_md(rows: list[Row]) -> str:
    lines = ["# Сравнение моделей на запросах по областям датасета", ""]
    if not rows:
        return "\n".join([*lines, "Готовых прогонов пока нет.", ""])
    if rows[0].matched_top is None:
        lines += ["Столбцы «совпало» пусты: нет data/*.xlsx или data/positive_terms.csv.", ""]
    lines += [
        "## Итог по моделям",
        "",
        "| Модель | Областей | Совпало в топ-15 | Совпало среди оценённых | В топе | Уверенных "
        "| Карточек без текста | Время, мин | Время LLM, мин |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in dict.fromkeys(r.model for r in rows):
        own = [r for r in rows if r.model == model]
        matched = None if own[0].matched_top is None else sum(r.matched_top or 0 for r in own)
        scored = None if own[0].matched_scored is None else sum(r.matched_scored or 0 for r in own)
        lines.append(
            f"| {model} | {len(own)} | {_n(matched)} | {_n(scored)} | {sum(r.top for r in own)} "
            f"| {sum(r.confident for r in own)} | {sum(r.no_text for r in own)} "
            f"| {sum(r.duration_s for r in own) / 60:.0f} | {sum(r.llm_s for r in own) / 60:.0f} |"
        )
    lines += [
        "",
        "## По областям",
        "",
        "| Модель | Область | Совпало в топ-15 | Документов | Кандидатов | В топе | Уверенных "
        "| Без текста | Вызовов LLM | Время, мин | Отказов источников |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.model} | {r.domain} | {_n(r.matched_top)} | {r.documents} | {r.candidates} | {r.top} "
        f"| {r.confident} | {r.no_text} | {r.llm_calls} | {r.duration_s / 60:.1f} | {r.failures} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def write_report() -> str:
    report = report_md(load_rows(reference=load_reference_if_present()))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def _child(spec: str, domains: list[str] | None) -> None:
    """Запустить одну модель отдельным процессом с её LLM_PROVIDER и LLM_MODEL в окружении."""
    provider, model = parse_spec(spec)
    env = {**os.environ, "LLM_PROVIDER": provider, "LLM_MODEL": model}
    cmd = [sys.executable, "-m", "src.pipeline.compare_models", "--one", spec]
    if domains:
        cmd += ["--domains", *domains]
    code = subprocess.run(cmd, env=env, check=False).returncode
    if code != 0:
        log.error("Модель %s завершилась с кодом %d — иду к следующей", spec, code)


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнить модели на запросах по областям датасета")
    ap.add_argument("models", nargs="*", help="провайдер:модель, например ollama:qwen3:4b gigachat:GigaChat-2")
    ap.add_argument("--domains", nargs="+", help="только эти области (по умолчанию все 6)")
    ap.add_argument("--report-only", action="store_true", help="только пересобрать отчёт по готовым прогонам")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args()

    all_domains = domain_queries()
    if args.domains and (unknown := set(args.domains) - set(all_domains)):
        ap.error(f"нет таких областей: {', '.join(sorted(unknown))}; есть: {', '.join(all_domains)}")
    if args.one:
        provider, model = parse_spec(args.one)
        domains = {d: q for d, q in all_domains.items() if not args.domains or d in args.domains}
        asyncio.run(run_model(provider, model, domains))
        return
    if not args.report_only:
        if not args.models:
            ap.error("укажи модели или --report-only")
        for spec in args.models:
            parse_spec(spec)  # опечатку в списке видно сразу, а не через час
        for spec in args.models:
            _child(spec, args.domains)
    print(write_report())
    print(f"Отчёт: {REPORT_PATH}; прогоны: {RUNS_DIR}")


if __name__ == "__main__":
    main()

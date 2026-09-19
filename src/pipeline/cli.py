"""Прогон из консоли: `python -m src.pipeline.cli "перспективные решения в финтехе"`.

Печатает результат по-русски: сколько собрано, что попало в топ, что исключено и почему,
какие модели вызывались. Нужен для промежуточной сдачи и для ежедневной проверки main.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.common.schemas import SearchResult, SignalCard
from src.pipeline.run import run

LINE = "─" * 78

# Метка «--json без имени файла»: печатаем JSON в stdout, человекочитаемый отчёт не печатаем.
_JSON_TO_STDOUT = "-"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    to_stdout = args.json is _JSON_TO_STDOUT
    progress = _show_progress if args.progress and not to_stdout else None
    result = asyncio.run(run(args.query, on_progress=progress))
    dump = result.model_dump_json(indent=2)
    if to_stdout:
        # JSON единственное, что уходит в stdout: так работает `--json > result.json`.
        print(dump)
        return 0 if result.status == "done" else 1
    if args.json:
        Path(args.json).write_text(dump, encoding="utf-8")
        print(f"\nРезультат целиком: {args.json}")
    _print_result(result)
    return 0 if result.status == "done" else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Поиск слабых сигналов по свободному запросу")
    parser.add_argument("query", help="запрос, например: перспективные решения в финтехе")
    parser.add_argument(
        "--json",
        nargs="?",
        const=_JSON_TO_STDOUT,
        metavar="ФАЙЛ",
        help="сохранить полный SearchResult в файл; без имени файла — напечатать JSON в stdout",
    )
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="не печатать шаги")
    parser.add_argument("--full", action="store_true", help="печатать все источники и причины")
    return parser.parse_args(argv)


def _show_progress(result: SearchResult) -> None:
    print(f"  [{result.stage}]", flush=True)


def _print_result(result: SearchResult) -> None:
    print(f"\n{LINE}\nЗапрос: {result.query}\nПрогон: {result.run_id}, статус: {result.status}")
    if result.error:
        print(f"Ошибка: {result.error}")
    print(
        f"Документов: {result.documents_processed} | кандидатов: {result.candidates_found} | "
        f"в выдаче: {len(result.top)} | уверенных: {result.confident_signals} | {result.duration_s} с"
    )
    if result.expanded_phrases:
        print(f"\nПоисковые фразы ({len(result.expanded_phrases)}): {'; '.join(result.expanded_phrases)}")
    print(f"\n{LINE}\nТОП СЛАБЫХ СИГНАЛОВ")
    for i, card in enumerate(result.top, 1):
        _print_card(i, card)
    if result.excluded:
        print(f"\n{LINE}\nИСКЛЮЧЕНО ({len(result.excluded)})")
        for decision in result.excluded:
            print(f"  — {decision.name} [{decision.reason_code}]: {decision.reason_text}")
    print(f"\n{LINE}\nВЫЗОВЫ МОДЕЛЕЙ")
    for call in result.model_calls:
        print(f"  {call.step}: {call.model} ({call.provider}), {call.duration_ms} мс")


def _print_card(number: int, card: SignalCard) -> None:
    title = card.name_ru or card.name
    print(f"\n{number:>2}. {title} — уверенность {card.score:.2f}")
    if card.name_ru:
        print(f"    оригинал: {card.name}")
    print(f"    {card.description}")
    print(f"    Преимущество: {card.advantage}")
    print(f"    Кейс: {card.case_example}")
    print(f"    Почему ранний сигнал: {card.why_weak_signal}")
    for reason in card.top_reasons:
        print(f"    • {reason.text} (вклад {reason.contribution:+.2f})")
    print(f"    Источники ({len(card.sources)}):")
    for src in card.sources:
        published = src.published.isoformat() if src.published else "дата неизвестна"
        print(f"      - {src.title} [{src.source_type.value}, {src.language}, доверие: {src.trust.value}]")
        print(f"        {src.url} ({published})")


if __name__ == "__main__":
    sys.exit(main())

"""Готовые прогоны на диске: data/search_runs/*.json.

Та же папка, куда пишут Streamlit (`{run_id}.json`) и проверка «как у жюри» (`<область>.json`), поэтому
имя файла не обязано совпадать с run_id — прогон ищется по содержимому. Файлы нужны стенду: база может
быть не поднята, а готовые прогоны должны пережить перезапуск сервера и открываться мгновенно.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from src.common.logs import get_logger
from src.common.schemas import SearchResult

log = get_logger(__name__)

RUNS_DIR = Path("data") / "search_runs"


class RunSummary(BaseModel):
    """Строка в списке прогонов: чтобы показать список, не нужно тянуть все карточки."""

    run_id: str
    query: str
    status: Literal["running", "done", "error"]
    stage: str
    started_at: datetime
    duration_s: float | None
    documents_processed: int
    candidates_found: int
    signals: int
    confident_signals: int


def summarize(result: SearchResult) -> RunSummary:
    return RunSummary(
        run_id=result.run_id,
        query=result.query,
        status=result.status,
        stage=result.stage,
        started_at=result.started_at,
        duration_s=result.duration_s,
        documents_processed=result.documents_processed,
        candidates_found=result.candidates_found,
        signals=len(result.top),
        confident_signals=result.confident_signals,
    )


def save(result: SearchResult, runs_dir: Path | None = None) -> None:
    """Записать прогон в файл. Не вышло — пишем в лог: выдача уже есть в памяти, прогон не теряется."""
    runs_dir = runs_dir or RUNS_DIR
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"{result.run_id}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning("Прогон %s не сохранён в %s: %s", result.run_id, runs_dir, exc)


def load_all(runs_dir: Path | None = None) -> list[SearchResult]:
    """Все читаемые прогоны из папки. Битый или чужой json пропускается с предупреждением."""
    runs_dir = runs_dir or RUNS_DIR
    if not runs_dir.is_dir():
        return []
    results = []
    for path in sorted(runs_dir.glob("*.json")):
        result = _read(path)
        if result is not None:
            results.append(result)
    return results


def find(run_id: str, runs_dir: Path | None = None) -> SearchResult | None:
    """Прогон по run_id: сначала файл с таким именем, потом все остальные."""
    runs_dir = runs_dir or RUNS_DIR
    direct = runs_dir / f"{run_id}.json"
    if direct.is_file():
        result = _read(direct)
        if result is not None and result.run_id == run_id:
            return result
    return next((r for r in load_all(runs_dir) if r.run_id == run_id), None)


def _read(path: Path) -> SearchResult | None:
    try:
        return SearchResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        log.warning("Файл прогона %s не прочитан: %s", path, exc)
        return None

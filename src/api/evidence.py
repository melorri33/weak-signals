"""Признаки и динамика публикаций кандидатов: data/search_evidence/{run_id}.json.

В SearchResult этого нет: контракт выдачи — карточки и оценки. А интерфейсу нужны измеренные числа,
чтобы показать, почему технология ранняя: карта «сколько публикаций — как быстро растут» по всем
кандидатам (включая отсеянных) и график публикаций по годам в инсайте. Отдельная папка, а не
data/search_runs: там каждый *.json читается как SearchResult.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.common.logs import get_logger
from src.common.schemas import Candidate, CandidateFeatures, TermStats

log = get_logger(__name__)

EVIDENCE_DIR = Path("data") / "search_evidence"


class CandidateEvidence(BaseModel):
    """Всё измеренное по одному кандидату. Годовые ряды пустые, если статистика не успела."""

    candidate_id: str
    name: str
    name_ru: str | None = None
    features: CandidateFeatures
    pubs_by_year: dict[int, int] = Field(default_factory=dict)
    patents_by_year: dict[int, int] | None = None
    news_by_year: dict[int, int] | None = None


class RunEvidence(BaseModel):
    run_id: str
    candidates: list[CandidateEvidence] = Field(default_factory=list)


def build(
    run_id: str,
    candidates: list[Candidate],
    features: list[CandidateFeatures],
    stats: dict[str, TermStats],
) -> RunEvidence:
    by_id = {c.id: c for c in candidates}
    items = []
    for feature in features:
        candidate = by_id.get(feature.candidate_id)
        if candidate is None:
            continue
        term = stats.get(feature.candidate_id)
        items.append(
            CandidateEvidence(
                candidate_id=candidate.id,
                name=candidate.name,
                name_ru=candidate.name_ru,
                features=feature,
                pubs_by_year=term.pubs_by_year if term else {},
                patents_by_year=term.patents_by_year if term else None,
                news_by_year=term.news_by_year if term else None,
            )
        )
    return RunEvidence(run_id=run_id, candidates=items)


def save(evidence: RunEvidence, evidence_dir: Path | None = None) -> None:
    """Записать в файл. Не вышло — только лог: выдача от этого не зависит."""
    evidence_dir = evidence_dir or EVIDENCE_DIR
    try:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / f"{evidence.run_id}.json").write_text(evidence.model_dump_json(), encoding="utf-8")
    except OSError as exc:
        log.warning("Признаки прогона %s не сохранены в %s: %s", evidence.run_id, evidence_dir, exc)


def load(run_id: str, evidence_dir: Path | None = None) -> RunEvidence | None:
    if not run_id.isalnum():  # run_id приходит из адреса и становится именем файла
        return None
    path = (evidence_dir or EVIDENCE_DIR) / f"{run_id}.json"
    if not path.is_file():
        return None
    try:
        return RunEvidence.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        log.warning("Файл признаков %s не прочитан: %s", path, exc)
        return None

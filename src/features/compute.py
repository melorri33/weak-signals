"""Признаки кандидата: `compute(candidate, docs, stats) -> CandidateFeatures`.

ЗАГЛУШКА. Сигнатура финальная, внутренности будут заменены. Сейчас считаются только признаки,
которые прямо выводятся из входа (без обращения к сети и моделям). Нет данных — поле None, не 0.
"""

from __future__ import annotations

from datetime import date

from src.common.schemas import Candidate, CandidateFeatures, Document, SourceType, TermStats

# Типы источников, которые считаем «наукой» для отношения новости/наука.
_SCIENCE_TYPES = {SourceType.PAPER, SourceType.PREPRINT, SourceType.PATENT}


def compute(candidate: Candidate, docs: list[Document], stats: TermStats | None) -> CandidateFeatures:
    """Посчитать признаки кандидата по его документам и статистике термина.

    docs можно передавать все документы прогона — берутся только те, что в candidate.document_ids.
    stats может быть None, если term_stats не отработал.
    """
    own_docs = _own_documents(candidate, docs)
    pubs = stats.pubs_by_year if stats else {}
    patents = stats.patents_by_year if stats else None
    news = stats.news_by_year if stats else None

    total_pubs = sum(pubs.values()) if pubs else None
    news_total = sum(news.values()) if news is not None else None

    return CandidateFeatures(
        candidate_id=candidate.id,
        total_pubs=total_pubs,
        growth_3y=_growth_3y(pubs),
        first_seen_year=min((y for y, n in pubs.items() if n > 0), default=None),
        patents_total=sum(patents.values()) if patents is not None else None,
        news_total=news_total,
        news_to_science_ratio=_news_to_science(news_total, total_pubs, own_docs),
        distinct_orgs=stats.distinct_orgs if stats else None,
        distinct_sources=len({d.source for d in own_docs}) if own_docs else None,
        has_wikipedia=_any_true(stats.wikipedia_ru, stats.wikipedia_en) if stats else None,
        has_standard=(stats.standard_mentions > 0) if stats and stats.standard_mentions is not None else None,
        stage=None,
    )


def _own_documents(candidate: Candidate, docs: list[Document]) -> list[Document]:
    ids = set(candidate.document_ids)
    return [d for d in docs if d.id in ids]


def _growth_3y(pubs_by_year: dict[int, int]) -> float | None:
    """Среднегодовой рост публикаций за 3 последних полных года (CAGR). None, если считать не из чего."""
    last = date.today().year - 1
    start = pubs_by_year.get(last - 3, 0)
    end = pubs_by_year.get(last, 0)
    if start <= 0:
        return None
    return round((end / start) ** (1 / 3) - 1, 4)


def _news_to_science(news_total: int | None, total_pubs: int | None, own_docs: list[Document]) -> float | None:
    """Сколько новостей на одну научную публикацию. Сначала по статистике, иначе по найденным документам."""
    if news_total is not None and total_pubs:
        return round(news_total / total_pubs, 4)
    science = sum(d.source_type in _SCIENCE_TYPES for d in own_docs)
    news_docs = sum(d.source_type == SourceType.NEWS for d in own_docs)
    if science == 0:
        return None
    return round(news_docs / science, 4)


def _any_true(*flags: bool | None) -> bool | None:
    if any(f is True for f in flags):
        return True
    if all(f is False for f in flags):
        return False
    return None

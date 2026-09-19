"""Признаки кандидата: `compute(candidate, docs, stats) -> CandidateFeatures`.

Одна функция для обоих этапов: для обучающей выборки и для открытого поиска признаки считаются
из одинаковых TermStats, поэтому модель видит в работе то же, на чём училась.
Нет данных — поле None, не 0.
"""

from __future__ import annotations

from datetime import date

from src.common.schemas import Candidate, CandidateFeatures, Document, SourceType, TermStats, TrustLevel

# Типы источников, которые считаем «наукой» для отношения новости/наука.
_SCIENCE_TYPES = {SourceType.PAPER, SourceType.PREPRINT, SourceType.PATENT}
# Год «появления» — первый год, когда публикаций набралось хотя бы столько (отсекает случайные совпадения слов).
_FIRST_SEEN_MIN_PUBS = 3


def last_full_year() -> int:
    """Последний закончившийся год — рост считаем по полным годам."""
    return date.today().year - 1


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
        first_seen_year=min((y for y, n in pubs.items() if n >= _FIRST_SEEN_MIN_PUBS), default=None),
        patents_total=sum(patents.values()) if patents is not None else None,
        news_total=news_total,
        news_to_science_ratio=_news_to_science(news_total, total_pubs, own_docs),
        distinct_orgs=stats.distinct_orgs if stats else None,
        distinct_sources=len({d.source for d in own_docs}) if own_docs else None,
        has_wikipedia=_any_true(stats.wikipedia_ru, stats.wikipedia_en) if stats else None,
        has_standard=(stats.standard_mentions > 0) if stats and stats.standard_mentions is not None else None,
        stage=None,
        extra=_extra(pubs, news, stats.pubs_by_type if stats else None) | _trust_counts(own_docs),
    )


def _trust_counts(own_docs: list[Document]) -> dict[str, float | None]:
    """Сколько документов кандидата с известным доверием и сколько из них средних/высоких.

    Нужно правилу шума (ТЗ): блоги, соцсети и пресс-релизы не могут быть единственным основанием.
    Доверие не проставлено ни у одного документа — None (правило не срабатывает).
    """
    rated = [d for d in own_docs if d.trust is not None]
    if not rated:
        return {"docs_rated": None, "docs_trusted": None}
    trusted = sum(d.trust in (TrustLevel.HIGH, TrustLevel.MEDIUM) for d in rated)
    return {"docs_rated": float(len(rated)), "docs_trusted": float(trusted)}


def _extra(
    pubs: dict[int, int], news: dict[int, int] | None, by_type: dict[str, int] | None
) -> dict[str, float | None]:
    """Динамика публикаций, внимание медиа и доля препринтов (у ранних технологий их много)."""
    last = last_full_year()
    total = sum(pubs.values())
    recent = sum(n for y, n in pubs.items() if y > last - 3)
    news_recent = sum(n for y, n in news.items() if y > last - 3) if news is not None else None
    return {
        "pubs_last_year": float(pubs.get(last, 0)) if pubs else None,
        "recent_share": round(recent / total, 4) if total else None,
        "news_recent": float(news_recent) if news_recent is not None else None,
        "preprint_share": _share(by_type, "preprint"),
    }


def _share(by_type: dict[str, int] | None, kind: str) -> float | None:
    total = sum(by_type.values()) if by_type else 0
    return round(by_type.get(kind, 0) / total, 4) if by_type and total else None


def _own_documents(candidate: Candidate, docs: list[Document]) -> list[Document]:
    ids = set(candidate.document_ids)
    return [d for d in docs if d.id in ids]


def _growth_3y(pubs_by_year: dict[int, int]) -> float | None:
    """Среднегодовой рост публикаций за 3 последних полных года (CAGR). None, если считать не из чего."""
    last = last_full_year()
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
    """True, если хоть один источник сказал «да»; False, если хоть один сказал «нет», а остальные молчат."""
    if any(f is True for f in flags):
        return True
    if any(f is False for f in flags):
        return False
    return None

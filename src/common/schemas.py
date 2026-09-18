"""Контракты данных между модулями проекта «Слабые сигналы».

Меняет только владелец ядра (ML). Остальные импортируют и используют как есть.
Если не хватает поля — опиши, что нужно, в PR или сообщении владельцу ядра.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Сколько сигналов в выдаче и с какой уверенности сигнал считается «уверенным» (для счётчика в UI).
TOP_N = 15
CONFIDENT_THRESHOLD = 0.75


def _now() -> datetime:
    return datetime.now(UTC)


class SourceType(str, Enum):
    """Тип источника. Показывается пользователю и влияет на уровень доверия."""

    PAPER = "paper"  # научная статья (журнал, конференция)
    PREPRINT = "preprint"  # arXiv и т.п.
    PATENT = "patent"
    NEWS = "news"  # СМИ
    REPORT = "report"  # аналитический отчёт
    GOV = "gov"  # госорганы, регуляторы, международные организации
    BLOG = "blog"
    SOCIAL = "social"
    PRESS_RELEASE = "press_release"
    OTHER = "other"


class TrustLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------- Данные: сбор источников ----------


class Document(BaseModel):
    """Один найденный документ (статья, патент, новость...)."""

    id: str = Field(description="'<source>:<внешний id>', например 'openalex:W4391234567'")
    source: str = Field(
        description="openalex | arxiv | semantic_scholar | patentsview | gdelt | cyberleninka | rss ..."
    )
    source_type: SourceType
    title: str
    abstract: str | None = None
    url: str
    published: date | None = None
    language: str = Field(default="en", description="ISO 639-1: 'ru', 'en', ...")
    authors: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    found_by: str = Field(default="", description="поисковая фраза, по которой документ найден")
    trust: TrustLevel | None = Field(default=None, description="проставляет trust.level_for")
    fetched_at: datetime = Field(default_factory=_now)


class TermStats(BaseModel):
    """Статистика по одному термину (кандидату) из открытых источников.

    None = данные получить не удалось (источник не ответил). Модель должна переживать пропуски.
    """

    term: str
    pubs_by_year: dict[int, int] = Field(default_factory=dict)
    patents_by_year: dict[int, int] | None = None
    news_by_year: dict[int, int] | None = None
    wikipedia_ru: bool | None = None
    wikipedia_en: bool | None = None
    standard_mentions: int | None = None
    distinct_orgs: int | None = None
    errors: list[str] = Field(default_factory=list, description="какие источники не ответили")


# ---------- Интеграция: кандидаты ----------


class Candidate(BaseModel):
    """Технология-кандидат, выделенная из найденных документов."""

    id: str = Field(description="slug, например 'zero-knowledge-kyc'")
    name: str = Field(description="каноническое название на языке оригинала")
    name_ru: str | None = None
    aliases: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)


# ---------- ML: признаки и скоринг ----------

Stage = Literal["research", "prototype", "product", "mass"]


class CandidateFeatures(BaseModel):
    """Признаки кандидата. Одинаково считаются для датасета и для открытого поиска."""

    candidate_id: str
    total_pubs: int | None = None
    growth_3y: float | None = Field(default=None, description="среднегодовой рост публикаций за 3 года")
    first_seen_year: int | None = None
    patents_total: int | None = None
    news_total: int | None = None
    news_to_science_ratio: float | None = None
    distinct_orgs: int | None = None
    distinct_sources: int | None = None
    has_wikipedia: bool | None = None
    has_standard: bool | None = None
    stage: Stage | None = None
    extra: dict[str, float | None] = Field(default_factory=dict, description="прочие признаки ML")


class FilterDecision(BaseModel):
    """Решение правил отсева по кандидату."""

    candidate_id: str
    name: str
    excluded: bool
    reason_code: Literal["mature", "hype", "noise", "ok"]
    reason_text: str = Field(
        description="по-русски, для интерфейса: «Зрелая технология: 45 000 публикаций, есть стандарт ISO»"
    )


class Explanation(BaseModel):
    """Один «ключевой предиктор» — вклад признака в решение модели."""

    feature: str
    value: float | str | None = None
    contribution: float = Field(description="вклад SHAP: >0 — за сигнал, <0 — против")
    text: str = Field(description="по-русски: «Публикации растут на 60% в год»")


class ScoredCandidate(BaseModel):
    candidate_id: str
    name: str
    score: float = Field(ge=0, le=1)
    top_reasons: list[Explanation] = Field(default_factory=list)


# ---------- Выдача ----------


class SourceRef(BaseModel):
    """Источник в карточке сигнала — всё, что ТЗ требует показать."""

    document_id: str
    title: str = Field(description="оригинальное название")
    url: str
    published: date | None = None
    source_type: SourceType
    language: str
    trust: TrustLevel
    ru_summary: str | None = Field(default=None, description="резюме на русском для зарубежных источников")
    summary_is_generated: bool = False
    is_machine_translated: bool = False


class SignalCard(BaseModel):
    """Карточка слабого сигнала в топ-15."""

    candidate_id: str
    name: str
    name_ru: str | None = None
    score: float = Field(ge=0, le=1)
    description: str
    advantage: str
    case_example: str
    why_weak_signal: str
    top_reasons: list[Explanation] = Field(default_factory=list)
    sources: list[SourceRef] = Field(min_length=1)
    report_md: str | None = Field(default=None, description="подробный отчёт, генерируется по клику")


class ModelCall(BaseModel):
    """Запись в лог моделей (требование ТЗ)."""

    step: str = Field(description="expand_query | extract_candidates | embeddings | score | make_card | translate ...")
    model: str
    provider: str = Field(description="ollama | huggingface | local | gigachat | yandexgpt ...")
    duration_ms: int
    ts: datetime = Field(default_factory=_now)


class SearchResult(BaseModel):
    """Результат прогона по запросу. Пока status='running', UI показывает stage как прогресс."""

    run_id: str
    query: str
    status: Literal["running", "done", "error"] = "running"
    stage: str = ""
    expanded_phrases: list[str] = Field(default_factory=list)
    documents_processed: int = 0
    candidates_found: int = 0
    confident_signals: int = Field(default=0, description=f"сколько сигналов со score > {CONFIDENT_THRESHOLD}")
    top: list[SignalCard] = Field(default_factory=list, max_length=TOP_N)
    excluded: list[FilterDecision] = Field(default_factory=list)
    model_calls: list[ModelCall] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_now)
    duration_s: float | None = None
    error: str | None = None

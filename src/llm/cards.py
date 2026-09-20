"""Карточки сигналов: `make_card(scored, docs) -> SignalCard`.

Тексты пишет LLM по промпту src/llm/prompts/make_card.md — строго по переданным документам.
Проверяем её ответ: id документов, которых не было во входе, выкидываем, а русские резюме
источников помечаем как сгенерированные (требование ТЗ — машинный текст должен быть виден).

Источники карточки собираются **только** из переданных документов, их модель не трогает:
ссылка, дата, тип, язык и уровень доверия берутся из самого документа. Если LLM недоступна,
карточка всё равно собирается — с честной пометкой, что описания нет.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.common.logs import get_logger
from src.common.schemas import Document, ScoredCandidate, SignalCard, SourceRef, TrustLevel
from src.llm.client import LLMClient, LLMError, dumps_ru
from src.llm.prompt_loader import render

log = get_logger(__name__)

# Сколько документов показываем в карточке (и отдаём модели в промпт).
MAX_SOURCES = 5
# Сколько знаков аннотации кладём в промпт: дальше идёт вода, а токены на процессоре дорогие.
ABSTRACT_CHARS = 600

NO_TEXT = "Описание не сформировано: языковая модель недоступна. Технологию оцените по источникам ниже."


class NoSourcesError(RuntimeError):
    """У кандидата не осталось ни одного проверенного источника — в выдачу он не идёт."""


class _SourceSummary(BaseModel):
    document_id: str = ""
    ru_summary: str = ""


class _CardText(BaseModel):
    """Схема ответа модели: только тексты, ничего проверяемого мы у неё не спрашиваем."""

    description: str = ""
    advantage: str = ""
    case_example: str = ""
    why_weak_signal: str = ""
    used_document_ids: list[str] = Field(default_factory=list)
    source_summaries: list[_SourceSummary] = Field(default_factory=list)


async def make_card(
    scored: ScoredCandidate,
    docs: list[Document],
    name_ru: str | None = None,
    client: LLMClient | None = None,
) -> SignalCard:
    """Собрать карточку сигнала по документам кандидата.

    docs — документы только этого кандидата. Если их нет, карточку не делаем: сигнал без
    источника в выдачу попасть не может.
    """
    chosen = docs[:MAX_SOURCES]
    if not chosen:
        raise NoSourcesError(f"{scored.candidate_id}: нет документов для карточки")

    text, summaries = await _ask_llm(scored, chosen, client)
    sources = [_source_ref(d, summaries.get(d.id)) for d in chosen]
    return SignalCard(
        candidate_id=scored.candidate_id,
        name=scored.name,
        name_ru=name_ru,
        score=scored.score,
        description=text.description or NO_TEXT,
        advantage=text.advantage or "Не описано в источниках.",
        case_example=text.case_example or "В документах применений не описано.",
        why_weak_signal=text.why_weak_signal or "Оценка основана на признаках, см. блок с вкладами.",
        top_reasons=scored.top_reasons,
        sources=sources,
    )


async def _ask_llm(
    scored: ScoredCandidate, docs: list[Document], client: LLMClient | None
) -> tuple[_CardText, dict[str, str]]:
    """Тексты карточки от модели. Отказ модели — не повод терять сигнал: вернём пустые тексты."""
    try:
        client = client or LLMClient.from_settings()
        answer = await client.ask_json(
            step="make_card",
            prompt=render("make_card", name=scored.name, documents=_documents_block(docs)),
            schema=_CardText,
            max_tokens=700,
        )
    except LLMError as exc:
        log.warning("make_card(%s): LLM не помогла (%s) — карточка без описания", scored.candidate_id, exc)
        return _CardText(), {}

    known = {d.id for d in docs}
    summaries = {}
    for item in answer.source_summaries:
        if item.document_id not in known:
            log.warning(
                "make_card(%s): резюме для чужого документа %s — выкидываю", scored.candidate_id, item.document_id
            )
            continue
        if item.ru_summary.strip():
            summaries[item.document_id] = " ".join(item.ru_summary.split())
    return answer, summaries


def _documents_block(docs: list[Document]) -> str:
    return "\n".join(
        dumps_ru(
            {
                "id": doc.id,
                "title": doc.title,
                "abstract": (doc.abstract or "")[:ABSTRACT_CHARS],
                "type": doc.source_type.value,
                "published": doc.published.isoformat() if doc.published else None,
            }
        )
        for doc in docs
    )


def keep_known_documents(document_ids: list[str], docs: list[Document]) -> list[Document]:
    """Оставить только те документы, которые действительно передавались модели.

    Так проверяем ответ LLM: id, которого не было во входе, — выдумка, выкидываем.
    """
    by_id = {d.id: d for d in docs}
    kept: list[Document] = []
    for doc_id in dict.fromkeys(document_ids):
        doc = by_id.get(doc_id)
        if doc is None:
            log.warning("LLM сослалась на документ %s, которого не было во входе — выкидываю", doc_id)
            continue
        kept.append(doc)
    return kept


def _source_ref(doc: Document, ru_summary: str | None) -> SourceRef:
    """Источник для карточки: все поля берём из документа, а не из текста модели."""
    return SourceRef(
        document_id=doc.id,
        title=doc.title,
        url=doc.url,
        published=doc.published,
        source_type=doc.source_type,
        language=doc.language,
        trust=doc.trust or TrustLevel.MEDIUM,
        ru_summary=ru_summary,
        # Резюме написала модель — по ТЗ это должно быть видно пользователю.
        summary_is_generated=ru_summary is not None,
        is_machine_translated=False,
    )

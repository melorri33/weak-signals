"""Карточки сигналов: `make_card(scored, docs) -> SignalCard`.

ЗАГЛУШКА текстов. Сигнатура финальная, тексты карточки пока шаблонные — их будет писать LLM
по промпту src/llm/prompts/make_card.md.

Что здесь уже настоящее и меняться не будет: источники карточки собираются **только** из
переданных документов (жёсткое требование ТЗ — никаких придуманных ссылок).
"""

from __future__ import annotations

from src.common.logs import get_logger
from src.common.schemas import Document, ScoredCandidate, SignalCard, SourceRef, TrustLevel

log = get_logger(__name__)

# Сколько документов показываем в карточке (и отдаём модели в промпт).
MAX_SOURCES = 5


class NoSourcesError(RuntimeError):
    """У кандидата не осталось ни одного проверенного источника — в выдачу он не идёт."""


async def make_card(scored: ScoredCandidate, docs: list[Document], name_ru: str | None = None) -> SignalCard:
    """Собрать карточку сигнала по документам кандидата.

    docs — документы только этого кандидата. Если их нет, карточку не делаем: сигнал без
    источника в выдачу попасть не может.
    """
    sources = [_source_ref(d) for d in docs[:MAX_SOURCES]]
    if not sources:
        raise NoSourcesError(f"{scored.candidate_id}: нет документов для карточки")
    return SignalCard(
        candidate_id=scored.candidate_id,
        name=scored.name,
        name_ru=name_ru,
        score=scored.score,
        description=f"ЗАГЛУШКА: описание технологии «{scored.name}» по {len(sources)} документам.",
        advantage="ЗАГЛУШКА: преимущество перед текущими решениями.",
        case_example="ЗАГЛУШКА: где уже пробуют применять.",
        why_weak_signal="ЗАГЛУШКА: почему это ранняя стадия, а не зрелая технология.",
        top_reasons=scored.top_reasons,
        sources=sources,
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


def _source_ref(doc: Document) -> SourceRef:
    """Источник для карточки: все поля берём из документа, а не из текста модели."""
    return SourceRef(
        document_id=doc.id,
        title=doc.title,
        url=doc.url,
        published=doc.published,
        source_type=doc.source_type,
        language=doc.language,
        trust=doc.trust or TrustLevel.MEDIUM,
        ru_summary=None,
        summary_is_generated=False,
        is_machine_translated=False,
    )

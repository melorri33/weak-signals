"""Выделение технологий-кандидатов: `extract_candidates(docs) -> list[Candidate]`.

Названия технологий из документов вытаскивает LLM (промпт src/llm/prompts/extract_candidates.md):
заголовок документа технологией не является — «Startup raises $12M to put AI on sensors» должно
стать «on-device inference», а не «Startup raises $12M».

Документы отдаются модели пачками, новости первыми: 71% источников датасета организаторов —
техноновости, и именно там описаны ранние технологии и раунды стартапов.

Если LLM недоступна или ответила негодным, работает запасной вариант — названия режутся из
заголовков по простым правилам. Качество заметно хуже, но прогон не падает (требование: отказ
одного шага не валит конвейер).
"""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field

from src.common.logs import get_logger
from src.common.schemas import Candidate, Document, SourceType
from src.llm.client import LLMClient, LLMError, dumps_ru
from src.llm.prompt_loader import render

log = get_logger(__name__)

MAX_CANDIDATES = 60
WORDS_IN_NAME = 3

# По name ищется статистика точной фразой («quantum sensing» в OpenAlex и техмедиа), а модель
# училась на терминах из 2–4 английских слов. Длинная придуманная фраза даст ноль публикаций,
# признаки окажутся пустыми и кандидат потеряется — поэтому предупреждаем в логе.
MAX_WORDS_IN_TERM = 5

# Сколько документов в одной пачке. Меньше пачка — короче ответ и меньше шансов, что модель
# начнёт пересказывать документы вместо выписывания терминов.
DOCS_IN_BATCH = 8
# Сколько документов вообще отдаём модели: на процессоре каждый вызов стоит десятки секунд.
MAX_DOCS_FOR_LLM = 32
# Сколько знаков аннотации кладём в промпт: дальше идёт вода, а токены на ноутбуке дорогие.
ABSTRACT_CHARS = 300
# Собственный бюджет шага. Держим его заметно ниже бюджета конвейера (CANDIDATES_BUDGET_S):
# отмена снаружи приходит посреди вызова модели и уносит всех уже выписанных кандидатов,
# поэтому останавливаемся сами и возвращаем то, что успели.
BUDGET_S = 300.0

# Служебные слова: с них название технологии не начинается и смысла не несут.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "based",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
    "using",
    "study",
    "towards",
    "и",
    "в",
    "во",
    "для",
    "на",
    "о",
    "об",
    "по",
    "при",
    "с",
    "со",
    "из",
    "как",
    "почему",
    "пока",
    "не",
}
_SPLIT_TITLE = re.compile(r"[:;,.—–()\[\]]")
_WORD = re.compile(r"[^\w\-+]+", re.UNICODE)

# Слишком общие названия: по ним находится обзор рынка, а не технология. Промпт их запрещает,
# но модель иногда всё равно их выписывает.
_TOO_BROAD = {
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "generative ai",
    "big data",
    "cloud computing",
    "blockchain",
    "cybersecurity",
    "robotics",
    "edge computing",
    "edge ai",
    "internet of things",
    "digital transformation",
    "искусственный интеллект",
    "машинное обучение",
}


class _Candidate(BaseModel):
    """Одна технология в ответе модели. document_ids — это метки d1, d2… из промпта."""

    name: str = ""
    name_ru: str | None = None
    aliases: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)


class _Answer(BaseModel):
    candidates: list[_Candidate] = Field(default_factory=list)


async def extract_candidates(docs: list[Document], client: LLMClient | None = None) -> list[Candidate]:
    """Выделить технологии-кандидаты из найденных документов.

    Один кандидат может опираться на несколько документов; кандидаты отсортированы по числу документов.
    """
    if not docs:
        return []
    try:
        client = client or LLMClient.from_settings()
        candidates = await _ask_llm(docs, client)
    except LLMError as exc:
        log.warning("extract_candidates: LLM не помогла (%s) — режу названия из заголовков", exc)
        candidates = []
    if not candidates:
        candidates = _from_titles(docs)
    candidates = sorted(candidates, key=lambda c: len(c.document_ids), reverse=True)[:MAX_CANDIDATES]
    warn_on_long_names(candidates)
    log.info("Кандидатов из %d документов: %d", len(docs), len(candidates))
    return candidates


async def _ask_llm(docs: list[Document], client: LLMClient) -> list[Candidate]:
    """Опросить модель по пачкам документов и склеить ответы."""
    chosen = _order_for_llm(docs)[:MAX_DOCS_FOR_LLM]
    batches = [chosen[i : i + DOCS_IN_BATCH] for i in range(0, len(chosen), DOCS_IN_BATCH)]
    merged: dict[str, Candidate] = {}
    started = time.perf_counter()
    slowest_batch_s = 0.0
    for number, batch in enumerate(batches, start=1):
        elapsed = time.perf_counter() - started
        # Начинаем пачку, только если она успеет закончиться: прерванный вызов модели ничего не даёт,
        # а время съедает. Ориентируемся на самую долгую из уже сделанных.
        if elapsed + slowest_batch_s > BUDGET_S:
            log.warning(
                "extract_candidates: бюджет %.0f с, прошло %.0f с — пачки с %d по %d не беру",
                BUDGET_S,
                elapsed,
                number,
                len(batches),
            )
            break
        batch_started = time.perf_counter()
        labels = {f"d{i}": doc for i, doc in enumerate(batch, start=1)}
        prompt = render("extract_candidates", documents=_documents_block(labels))
        try:
            answer = await client.ask_json(step="extract_candidates", prompt=prompt, schema=_Answer, max_tokens=500)
        except LLMError as exc:
            log.warning("extract_candidates: пачка %d из %d не удалась (%s) — иду дальше", number, len(batches), exc)
            continue
        slowest_batch_s = max(slowest_batch_s, time.perf_counter() - batch_started)
        _merge(merged, answer.candidates, labels)
    if not merged:
        raise LLMError("ни одна пачка документов не дала кандидатов")
    return list(merged.values())


def _order_for_llm(docs: list[Document]) -> list[Document]:
    """Сначала новости, потом наука: у организаторов 71% источников — техноновости.

    Внутри группы свежие идут первыми, документы без даты — последними.
    """

    def key(doc: Document) -> tuple[int, int]:
        news_first = 0 if doc.source_type == SourceType.NEWS else 1
        recent_first = -doc.published.toordinal() if doc.published else 0
        return (news_first, recent_first)

    return sorted(docs, key=key)


def _documents_block(labels: dict[str, Document]) -> str:
    """Документы для промпта: короткие метки вместо длинных id — они занимают меньше токенов."""
    return "\n".join(
        dumps_ru(
            {
                "id": label,
                "title": doc.title,
                "abstract": (doc.abstract or "")[:ABSTRACT_CHARS],
                "type": doc.source_type.value,
            }
        )
        for label, doc in labels.items()
    )


def _merge(merged: dict[str, Candidate], found: list[_Candidate], labels: dict[str, Document]) -> None:
    """Добавить кандидатов пачки к общему списку, склеивая одинаковые по slug."""
    for item in found:
        name = " ".join(item.name.split())
        if not _is_usable(name):
            continue
        doc_ids = [labels[label].id for label in dict.fromkeys(item.document_ids) if label in labels]
        if not doc_ids:
            # Модель не указала ни одного документа из пачки — брать такое нельзя:
            # карточка собирается только по документам кандидата.
            log.info("extract_candidates: «%s» без документов из пачки — пропускаю", name)
            continue
        key = _slug(name)
        candidate = merged.get(key)
        if candidate is None:
            merged[key] = Candidate(
                id=key,
                name=name,
                name_ru=(item.name_ru or "").strip() or None,
                aliases=[a for a in item.aliases if a.strip()],
                document_ids=doc_ids,
            )
            continue
        candidate.document_ids.extend(d for d in doc_ids if d not in candidate.document_ids)
        candidate.aliases.extend(a for a in item.aliases if a.strip() and a not in candidate.aliases)
        if candidate.name_ru is None and item.name_ru:
            candidate.name_ru = item.name_ru.strip() or None


def _is_usable(name: str) -> bool:
    """Отсеять пустое, слишком общее и слишком длинное — такое кандидатом быть не может."""
    words = name.split()
    if not (1 <= len(words) <= 8):
        return False
    if name.lower() in _TOO_BROAD:
        log.info("extract_candidates: «%s» — слишком широкая область, пропускаю", name)
        return False
    return any(ch.isalpha() for ch in name)


def _from_titles(docs: list[Document]) -> list[Candidate]:
    """Запасной вариант без LLM: название технологии режется из заголовка документа."""
    by_key: dict[str, Candidate] = {}
    for doc in docs:
        name = _name_from_title(doc.title)
        if not name:
            continue
        key = _slug(name)
        candidate = by_key.get(key)
        if candidate is None:
            by_key[key] = Candidate(id=key, name=name, document_ids=[doc.id])
        elif doc.id not in candidate.document_ids:
            candidate.document_ids.append(doc.id)
    return list(by_key.values())


def warn_on_long_names(candidates: list[Candidate]) -> list[Candidate]:
    """Предупредить о названиях, по которым статистику точной фразой найти не получится.

    Кандидатов не выбрасываем: статистика — не единственный источник признаков, а терять технологию
    из-за формы названия хуже, чем считать её признаки по найденным документам.
    """
    long_names = [c.name for c in candidates if len(c.name.split()) > MAX_WORDS_IN_TERM]
    if long_names:
        log.warning(
            "Названия длиннее %d слов — статистика по точной фразе будет пустой (%d из %d): %s",
            MAX_WORDS_IN_TERM,
            len(long_names),
            len(candidates),
            "; ".join(long_names[:5]),
        )
    return candidates


def _name_from_title(title: str) -> str:
    """Название технологии из названия документа: первая смысловая часть, до WORDS_IN_NAME слов."""
    head = _SPLIT_TITLE.split(title.strip())[0]
    words = [w for w in head.split() if _WORD.sub("", w.lower()) not in _STOPWORDS]
    return " ".join(words[:WORDS_IN_NAME])


def _slug(name: str) -> str:
    """Ключ склейки и id кандидата: 'Zero-knowledge KYC verification' → 'zero-knowledge-kyc-verification'."""
    parts = [_WORD.sub("", w.lower()) for w in name.split()]
    return "-".join(p for p in parts if p)


__all__ = ["extract_candidates", "warn_on_long_names"]

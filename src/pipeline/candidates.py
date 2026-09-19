"""Выделение технологий-кандидатов: `extract_candidates(docs) -> list[Candidate]`.

ЗАГЛУШКА. Сигнатура финальная. Сейчас кандидаты вырезаются из названий документов по простым
правилам — без LLM и эмбеддингов. Настоящая версия (промпт src/llm/prompts/extract_candidates.md):
аннотации пачками по 20 → LLM отдаёт названия технологий и id документов → склейка близких по
эмбеддингам → 40–60 кандидатов.
"""

from __future__ import annotations

import re

from src.common.logs import get_logger
from src.common.schemas import Candidate, Document

log = get_logger(__name__)

MAX_CANDIDATES = 60
WORDS_IN_NAME = 3

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


async def extract_candidates(docs: list[Document]) -> list[Candidate]:
    """Выделить технологии-кандидаты из найденных документов.

    Один кандидат может опираться на несколько документов; документы без внятного названия
    пропускаются. Кандидаты отсортированы по числу документов.
    """
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
    candidates = sorted(by_key.values(), key=lambda c: len(c.document_ids), reverse=True)[:MAX_CANDIDATES]
    log.info("Кандидатов из %d документов: %d", len(docs), len(candidates))
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

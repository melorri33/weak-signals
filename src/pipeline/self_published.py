"""Самопубликации без рецензии — не научный источник: убираем их сразу после сбора.

OpenAlex индексирует всё, у чего есть DOI, в том числе хранилища, куда любой загружает что угодно:
Zenodo, ResearchGate, Figshare. В выдаче такие записи выглядят как обычные статьи (source_type=paper).
Прогон 26.09 на GigaChat 2 Lite с MAX_DOCUMENTS=4000: в топ-15 по шести областям 11 карточек
оказались околонаучным мусором («онтологические теории», «теория» ускорения RSA-ключей и т.п.),
и 38 из 41 их источника — Zenodo и ResearchGate. Уверенность у них доходила до 81% — выше,
чем у настоящих сигналов.

SSRN и arXiv не трогаем: это препринт-серверы с модерацией, а SSRN — главный источник препринтов
по финансам.
"""

from __future__ import annotations

from urllib.parse import urlparse

from src.common.logs import get_logger
from src.common.schemas import Document

log = get_logger(__name__)

# Префиксы DOI хранилищ самопубликаций.
SELF_PUBLISHED_DOI_PREFIXES = {
    "10.5281",  # Zenodo
    "10.13140",  # ResearchGate
    "10.6084",  # Figshare
}
# Те же хранилища, если в ссылке их сайт, а не DOI.
SELF_PUBLISHED_HOSTS = {"zenodo.org", "researchgate.net", "figshare.com"}


def is_self_published(doc: Document) -> bool:
    """Документ из хранилища самопубликаций (по DOI или по адресу сайта)."""
    parsed = urlparse(doc.url)
    host = parsed.netloc.lower().removeprefix("www.")
    if host in {"doi.org", "dx.doi.org"}:
        prefix = parsed.path.lstrip("/").split("/", 1)[0]
        return prefix in SELF_PUBLISHED_DOI_PREFIXES
    return any(host == site or host.endswith(f".{site}") for site in SELF_PUBLISHED_HOSTS)


def without_self_published(docs: list[Document]) -> list[Document]:
    """Документы без самопубликаций; сколько убрали — в лог."""
    kept = [doc for doc in docs if not is_self_published(doc)]
    if dropped := len(docs) - len(kept):
        log.info("Убрано самопубликаций без рецензии (Zenodo, ResearchGate, Figshare): %d из %d", dropped, len(docs))
    return kept

"""OpenAlex: научные статьи. Поиск документов для collect(), счётчики для term_stats().

Точная фраза обязательна — без кавычек OpenAlex ищет слова по отдельности и раздувает выдачу
на порядки («quantum sensing»: 62 150 работ вместо 7 113, живой запрос 19.09.2026).
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import httpx

from src.common.config import Settings
from src.common.logs import get_logger
from src.common.schemas import Document, SourceType

log = get_logger(__name__)

API = "https://api.openalex.org/works"
# OpenAlex не отдаёт больше 200 групп на group_by — используем как верхнюю границу и для per_page.
MAX_PER_PAGE = 200
# Для collect(): старые обзоры и учебники по устоявшимся темам иначе лезут наверх выдачи и
# вытесняют свежие документы — term_stats считает историю целиком, туда этот фильтр не идёт.
SEARCH_YEARS_BACK = 3


def _params(term: str, settings: Settings, extra_filter: str | None = None, **extra: str) -> dict[str, str]:
    filter_value = f'title_and_abstract.search:"{term}"'
    if extra_filter:
        filter_value += f",{extra_filter}"
    params = {"filter": filter_value, **extra}
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key
    elif settings.contact_email:
        params["mailto"] = settings.contact_email
    return params


def _rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex отдаёт аннотацию как {слово: [позиции]} — собираем обратно в текст."""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _to_document(work: dict, phrase: str) -> Document:
    doi = work.get("doi")
    url = doi or (work.get("open_access") or {}).get("oa_url") or work["id"]
    authors = [
        a["author"]["display_name"] for a in work.get("authorships", []) if a.get("author", {}).get("display_name")
    ]
    orgs: list[str] = []
    for a in work.get("authorships", []):
        for inst in a.get("institutions", []) or []:
            name = inst.get("display_name")
            if name and name not in orgs:
                orgs.append(name)
    published = None
    if pub_date := work.get("publication_date"):
        try:
            published = date.fromisoformat(pub_date)
        except ValueError:
            published = None
    return Document(
        id=f"openalex:{re.sub(r'^https://openalex.org/', '', work['id'])}",
        source="openalex",
        source_type=SourceType.PAPER,
        title=work.get("title") or work.get("display_name") or "",
        abstract=_rebuild_abstract(work.get("abstract_inverted_index")),
        url=url,
        published=published,
        language=work.get("language") or "en",
        authors=authors,
        organizations=orgs,
        found_by=phrase,
    )


async def _get(params: dict[str, str], settings: Settings, client: httpx.AsyncClient) -> dict:
    """Запрос к OpenAlex. Ключ исчерпан — повторяем без него, анонимный доступ живёт отдельно.

    У ключа свой дневной лимит (около $1 против $0.10 у анонимного доступа). Когда он выбран,
    OpenAlex отвечает 429 именно на запросы с ключом, а без ключа в это же время отдаёт данные.
    Проверено 21.09: с ключом 429, без ключа HTTP 200 и 7116 работ по той же фразе.

    Без отката конвейер терял научный источник целиком и молча шёл дальше: в одном из прогонов
    того дня 1372 запроса из 1380 отбились, три четверти кандидатов остались без данных
    о публикациях, а в логе прогона это выглядело как обычная работа.
    """
    r = await client.get(API, params=params, timeout=settings.source_timeout_s)
    if r.status_code == 429 and "api_key" in params:
        log.warning("OpenAlex: ключ исчерпал лимит, повторяю запрос без него")
        retry = {k: v for k, v in params.items() if k != "api_key"}
        if settings.contact_email:
            retry["mailto"] = settings.contact_email
        r = await client.get(API, params=retry, timeout=settings.source_timeout_s)
    r.raise_for_status()
    return r.json()


async def search(phrase: str, settings: Settings, client: httpx.AsyncClient, limit: int = 200) -> list[Document]:
    """Документы, у которых фраза встречается в заголовке или аннотации, за последние годы."""
    from_date = date.today() - timedelta(days=365 * SEARCH_YEARS_BACK)
    params = _params(
        phrase,
        settings,
        extra_filter=f"from_publication_date:{from_date.isoformat()}",
        per_page=str(min(limit, MAX_PER_PAGE)),
        select="id,doi,title,display_name,abstract_inverted_index,publication_date,language,authorships,open_access",
    )
    data = await _get(params, settings, client)
    return [_to_document(work, phrase) for work in data["results"]]


async def year_counts(term: str, settings: Settings, client: httpx.AsyncClient) -> dict[int, int]:
    """Число работ по годам публикации (для TermStats.pubs_by_year)."""
    params = _params(term, settings, group_by="publication_year", per_page=str(MAX_PER_PAGE))
    data = await _get(params, settings, client)
    return {int(g["key"]): g["count"] for g in data["group_by"] if g["key"].isdigit()}


async def type_counts(term: str, settings: Settings, client: httpx.AsyncClient) -> dict[str, int]:
    """Число работ по типу публикации (для TermStats.pubs_by_type — доля препринтов)."""
    params = _params(term, settings, group_by="type", per_page=str(MAX_PER_PAGE))
    data = await _get(params, settings, client)
    return {g["key_display_name"]: g["count"] for g in data["group_by"]}


async def org_count(term: str, settings: Settings, client: httpx.AsyncClient) -> int:
    """Число разных организаций-авторов (для TermStats.distinct_orgs). OpenAlex отдаёт не больше 200 групп."""
    params = _params(term, settings, group_by="authorships.institutions.lineage", per_page=str(MAX_PER_PAGE))
    data = await _get(params, settings, client)
    return int(data["meta"]["groups_count"])

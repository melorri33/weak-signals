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

# Сколько кандидатов передаём дальше.
#
# Сейчас этот предел не срабатывает: шаг останавливается раньше по своему BUDGET_S, успевая
# около двенадцати пачек, и кандидатов выходит 47-65 — прежний предел 60 стоял ровно там, где
# шаг и так заканчивается. Подъём до 150 на прогоне 21.09 ничего не изменил, это проверено.
#
# Поднят на будущее: при большем BUDGET_S шаг находит 132-234 технологии (замер на трёх
# областях), и тогда предел начал бы резать до трёх четвертей из них молча.
#
# Порядок отсечения — по времени появления, а не по числу документов. Сортировка по упоминаемости
# была прямо против задачи: замер 21.09 показал, что из 12 технологий датасета, названных
# в собранных документах, 9 встретились ровно в одном документе. Такие кандидаты оказывались
# в самом хвосте и срезались первыми, а наверху оставались самые упоминаемые — то есть зрелые.
# Пока предел не упирался, это ничего не портило; после ускорения чтения кандидатов стало ровно 150.
#
# Выше 150 не ставим: каждый кандидат стоит запроса статистики.
MAX_CANDIDATES = 150
WORDS_IN_NAME = 3

# По name ищется статистика точной фразой («quantum sensing» в OpenAlex и техмедиа), а модель
# училась на терминах из 2–4 английских слов. Длинная придуманная фраза даст ноль публикаций,
# признаки окажутся пустыми и кандидат потеряется — поэтому предупреждаем в логе.
MAX_WORDS_IN_TERM = 5

# Сколько документов в одной пачке. Меньше пачка — короче ответ и меньше шансов, что модель
# начнёт пересказывать документы вместо выписывания терминов.
DOCS_IN_BATCH = 8
# Сколько документов вообще отдаём модели. На видеокарте вызов стоит секунды, поэтому отдаём всё,
# что собрали (MAX_DOCUMENTS в .env): чем больше документов посмотрела модель, тем больше шансов,
# что нужная технология вообще попадёт в кандидаты. Реальную границу ставит BUDGET_S ниже.
MAX_DOCS_FOR_LLM = 500
# Сколько знаков аннотации кладём в промпт: дальше идёт вода, а токены на ноутбуке дорогие.
ABSTRACT_CHARS = 300
# Собственный бюджет шага. Держим его заметно ниже бюджета конвейера (CANDIDATES_BUDGET_S):
# отмена снаружи приходит посреди вызова модели и уносит всех уже выписанных кандидатов,
# поэтому останавливаемся сами и возвращаем то, что успели.
BUDGET_S = 150.0

# Общие слова, которыми модель подменяет название технологии: «edge ai infrastructure» вместо
# названия самого чипа, «ai agent certification» вместо технологии, которую сертифицируют. Такие записи
# занимают место в топ-15 и плодят дубликаты — четыре слота на варианты одного чипа.
# Проверяем только последнее слово: оно говорит, чем технология является.
#
# Список сверен с data/positive_terms.csv: ни один термин датасета организаторов на эти слова
# не оканчивается. Слово «platform» намеренно не включено: в датасете есть термин с таким
# последним словом, и запрет отсёк бы настоящее попадание. Прежде чем дополнять список,
# прогони сверку заново — правило не должно отсекать правильные ответы.
# Единственное и множественное число перечислены парами: «platforms» здесь было,
# а «platform» не было, и записи вроде «… platform» спокойно проходили фильтр и занимали
# места в топ-15. Замер 21.09 по двум прогонам: таких записей 5 и 8 из 90 мест.
_TOO_GENERIC_HEAD = {
    "capabilities",
    "capability",
    "certification",
    "certifications",
    "component",
    "components",
    "deployment",
    "deployments",
    "ecosystem",
    "ecosystems",
    "foundation",
    "foundations",
    "framework",
    "frameworks",
    "infrastructure",
    "infrastructures",
    "offering",
    "offerings",
    "platform",
    "platforms",
    "portfolio",
    "portfolios",
    "processing",
    "product",
    "products",
    "service",
    "services",
    "solution",
    "solutions",
    "stack",
    "stacks",
    "suite",
    "suites",
    "system",
    "systems",
    "technologies",
    "technology",
    "tool",
    "tools",
}

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
# Номер версии или модели («Foo 3.8», «Bar 2») — признак продукта, а не технологии.
_VERSION_NUMBER = re.compile(r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])")
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
    """Одна технология в ответе модели. document_ids — это метки d1, d2… из промпта.

    Русского названия и синонимов здесь нет намеренно. Генерация ответа — 85-90% времени
    шага (замер 21.09 по таймингам Ollama), а эти два поля занимали половину ответа при том,
    что конвейеру не нужны: синонимы не читает никто, кроме диагностики, а русское название
    нужно пятнадцати карточкам из полутора сотен кандидатов — его пишет make_card.
    Замер: 4.0 с на пачку вместо 9.0, то есть 120 документов в минуту вместо 53.
    """

    name: str = ""
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
    candidates = candidates[:MAX_CANDIDATES]
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


# На сколько новостей приходится один научный документ в очереди к модели. Новостей больше,
# потому что у организаторов 71% источников датасета — техноновости. Но не все: наука находит
# технологии, которых в новостях нет (замер 21.09: arXiv находит 69 технологий датасета из 100,
# новостные ленты — 63).
NEWS_PER_PAPER = 1


def _order_for_llm(docs: list[Document]) -> list[Document]:
    """Новости и наука вперемежку, один к одному.

    Раньше очередь была строгой — сначала все новости, потом вся наука. При этом шаг успевает
    прочитать около сотни документов из тысячи с лишним, поэтому до науки очередь не доходила
    никогда: её собирали, тратили на это время и бюджет OpenAlex, и выбрасывали не глядя.
    Внутри новостей очередь занимала одна лента: в прочитанной сотне 81 документ приходился
    на один сайт.

    Замер 21.09 на корпусе из шести областей. В собранных документах названы 12 технологий
    датасета; считали на глубине 112 документов — именно столько шаг успевает прочитать:

        сначала новости, потом наука    2 из 12
        две новости на одну статью      4 из 12
        одна новость на одну статью     5 из 12

    Проверяли и другое: обход источников по кругу, потолок на число документов с одного сайта,
    склейку одинаковых заголовков, подъём новостей про раунды стартапов. Ни один из этих
    способов не дал прироста сверх простого чередования.

    Внутри каждой группы свежие идут первыми, документы без даты — последними.
    """

    def key(doc: Document) -> int:
        return -doc.published.toordinal() if doc.published else 0

    news = sorted((d for d in docs if d.source_type == SourceType.NEWS), key=key)
    papers = sorted((d for d in docs if d.source_type != SourceType.NEWS), key=key)
    mixed: list[Document] = []
    n = p = 0
    while n < len(news) or p < len(papers):
        mixed.extend(news[n : n + NEWS_PER_PAPER])
        n += NEWS_PER_PAPER
        mixed.extend(papers[p : p + 1])
        p += 1
    return mixed


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
            merged[key] = Candidate(id=key, name=name, document_ids=doc_ids)
            continue
        candidate.document_ids.extend(d for d in doc_ids if d not in candidate.document_ids)


# Длиннее этого название перестаёт быть термином и становится описанием. Замер 21.09
# по прогону из шести областей: у названий в 2-3 слова статистика публикаций находится
# в 70-94% случаев, у четырёхсловных — в 29%, у пятисловных и длиннее — в 14%. Средняя
# оценка таких кандидатов 0.06, и ни один из них не попал в топ-15: по фразе, которой
# никто не пишет, публикаций не найти, и модель ранжирует их вслепую.
MAX_NAME_WORDS = 4


def _is_usable(name: str) -> bool:
    """Отсеять пустое, слишком общее и похожее на название продукта — такое кандидатом быть не может."""
    words = name.split()
    if not (1 <= len(words) <= MAX_NAME_WORDS):
        log.info("extract_candidates: «%s» — это описание, а не термин, пропускаю", name)
        return False
    if name.lower() in _TOO_BROAD:
        log.info("extract_candidates: «%s» — слишком широкая область, пропускаю", name)
        return False
    if _looks_like_product(name):
        log.info("extract_candidates: «%s» похоже на название продукта, а не технологии — пропускаю", name)
        return False
    if _has_generic_head(name):
        log.info("extract_candidates: «%s» — общее слово вместо технологии, пропускаю", name)
        return False
    return any(ch.isalpha() for ch in name)


def _looks_like_product(name: str) -> bool:
    """«Gemini 3.8 Flash», «Voyager Wingman», «Cato SASE Platform» — это продукты, а не технологии.

    На живом прогоне модель 4B выписывала из новостей ровно их: 15 кандидатов из 16 были названиями
    моделей, устройств и платформ. Промпт требует писать термин строчными буквами (заглавные — только
    в аббревиатурах), поэтому имя собственное в середине названия и номер версии — надёжные признаки.
    """
    if _VERSION_NUMBER.search(name):
        return True
    return any(word[:1].isupper() and not word.isupper() for word in name.split()[1:])


def _has_generic_head(name: str) -> bool:
    """«edge ai infrastructure», «ai agent certification» — сказано про технологию, но не названа она сама.

    Смотрим последнее слово: оно отвечает на вопрос, чем технология является. Если это инфраструктура,
    решение, сервис или процесс — записи в выдаче не место: по ней не найти публикации точной фразой,
    а место в топ-15 она займёт.
    """
    words = name.split()
    return bool(words) and words[-1].strip(",.").lower() in _TOO_GENERIC_HEAD


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

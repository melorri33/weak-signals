"""Близость к теме запроса и почти-дубли — поправки к ранжированию по эмбеддингам bge-m3.

Модели по цифрам публикаций и по названию оценивают, «похоже ли на раннюю технологию», но не знают,
о чём спросили: тему проверяет только фильтр по верхним 45 (да/нет). Поэтому в топ-15 попадали методы
из чужих статей, а технологии самой области оставались ниже. Тема — центр эмбеддингов запроса и фраз,
которые модель придумала для поиска (без фраз свежести: они одинаковы во всех областях).
Поправка — β·(близость − медиана по кандидатам запроса): выше медианы — плюс, ниже — минус.

Замер 28.09 на четырёх сохранённых прогонах GigaChat (24 области): технологий датасета в топ-15
20 → 23 при β = 0.3; технологии датасета по близости к теме выше 81% кандидатов своей области.
Из топа уходили методы и материалы из статей, приходили технологии области.
"""

from __future__ import annotations

import numpy as np

from src.common.config import get_settings
from src.common.logs import get_logger, model_timer
from src.common.phrases import is_fresh_phrase
from src.common.schemas import Explanation, ScoredCandidate
from src.model import names

log = get_logger(__name__)


def with_topic_relevance(scored: list[ScoredCandidate], query: str, phrases: list[str]) -> list[ScoredCandidate]:
    """Оценки с поправкой на близость к теме запроса, по убыванию. Нет эмбеддингов — список как был."""
    weight = get_settings().topic_relevance_weight
    if weight <= 0 or len(scored) < 2:
        return scored
    topic = [p for p in phrases if not is_fresh_phrase(p)] + [query]
    vectors = _embed([*topic, *(s.name for s in scored)], step="topic_relevance")
    if vectors is None:
        return scored
    center = vectors[: len(topic)].mean(axis=0)
    closeness = vectors[len(topic) :] @ (center / np.linalg.norm(center))
    median = float(np.median(closeness))
    out = []
    for item, value in zip(scored, closeness, strict=True):
        delta = weight * (float(value) - median)
        reasons = item.top_reasons
        # В причины — только довод «за»: прогон 28.09 показал, что «дальше от темы» у верхних карточек
        # сбивает с толку (у них высокая оценка по другим причинам), а поправка к оценке и так учтена.
        if delta > 0:
            reason = Explanation(
                feature="topic_relevance",
                value=round(float(value), 4),
                contribution=round(delta, 4),
                text="Близко к теме запроса",
            )
            reasons = [*item.top_reasons[:1], reason, *item.top_reasons[1:]]
        out.append(
            item.model_copy(update={"score": round(min(max(item.score + delta, 0.0), 1.0), 4), "top_reasons": reasons})
        )
    return sorted(out, key=lambda s: s.score, reverse=True)


def drop_near_duplicates(scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Убрать кандидатов, чьё название почти совпадает по смыслу с названием выше по списку.

    Прогон 28.09: в топ-15 защиты ИИ шли подряд «semantic watermarking» и «neural network watermarking»,
    в робототехнике — «soft artificial muscle» и «artificial soft muscles»: для жюри это один сигнал,
    а место под другой технологией занято. Порог — NEAR_DUPLICATE_THRESHOLD (косинус bge-m3).
    """
    threshold = get_settings().near_duplicate_threshold
    if threshold >= 1 or len(scored) < 2:
        return scored
    vectors = _embed([s.name for s in scored], step="near_duplicates")
    if vectors is None:
        return scored
    kept: list[int] = []
    for i in range(len(scored)):
        twin = next((k for k in kept if float(vectors[k] @ vectors[i]) >= threshold), None)
        if twin is None:
            kept.append(i)
        else:
            log.info("Почти-дубль: «%s» пропускаю, уже есть «%s»", scored[i].name, scored[twin].name)
    return [scored[i] for i in kept]


def _embed(texts: list[str], step: str) -> np.ndarray | None:
    """Эмбеддинги тем же кодировщиком, что у модели названий; нет модели или упал — None."""
    art = names.artifact()
    if art is None:
        return None
    try:
        with model_timer(step=step, model=art["encoder"], provider=names.PROVIDER):
            return names.embed(texts, art["encoder"])
    except Exception:
        log.exception("%s: эмбеддинги не посчитались — порядок не меняю", step)
        return None

"""Проверка, что кандидат — технология, а не метод из статьи и не зрелый стандарт.

Прогоны 27–28.09 на GigaChat 2 Pro и Max: около половины мусора в топ-15 — методы и понятия из
научных статей («усечение ряда Тейлора», «шары спектральной нормы», гибрид LSTM и трансформера),
ещё пятая часть — давно массовые стандарты (AES-256, SHA-256). Их приносит научный поиск, а
модель выделения кандидатов записывает всё, что в статье названо. Такие кандидаты уходят в
отсеянные с причиной, как требует ТЗ: «зрелые, стандарты, шум исключаются с указанием причины».
"""

from typing import Literal

from pydantic import BaseModel

from src.common.logs import get_logger
from src.common.schemas import FilterDecision, ScoredCandidate
from src.llm.client import LLMClient, LLMError
from src.llm.prompt_loader import render
from src.pipeline.topic import CHECKED, MAX_TOKENS

log = get_logger(__name__)

# Больше этой доли — ответу не верим. Предел выше, чем у проверки темы: в слабых областях методов
# и понятий бывает до трети верхних кандидатов (Max 28.09, индустриальный ИИ: 10 из 15 карточек — мусор).
MAX_SHARE = 1 / 2
_REASONS: dict[str, tuple[Literal["mature", "noise"], str]] = {
    "method": ("noise", "Не технология: научный метод, алгоритм или понятие"),
    "standard": ("mature", "Зрелый стандарт или массовый продукт"),
}


class _NotTechnology(BaseModel):
    id: str
    kind: str = ""
    # Самопроверка: не зарождающаяся ли это всё-таки технология, которую уже делают компании.
    # Нет ответа — считаем технологией и не отсеиваем.
    is_emerging_technology: bool = True


class _Answer(BaseModel):
    not_technology: list[_NotTechnology]


async def not_technology(
    query: str, scored: list[ScoredCandidate], client: LLMClient | None = None
) -> list[FilterDecision]:
    """Решения об отсеве для верхних кандидатов, которые не технология. Модель не ответила — никого."""
    checked = scored[:CHECKED]
    if not checked:
        return []
    labels = {f"t{i}": item for i, item in enumerate(checked, start=1)}
    prompt = render(
        "check_technology", query=query, candidates="\n".join(f"- {k}: {v.name}" for k, v in labels.items())
    )
    try:
        client = client or LLMClient.from_settings()
        answer = await client.ask_json(step="check_technology", prompt=prompt, schema=_Answer, max_tokens=MAX_TOKENS)
    except LLMError as exc:
        log.warning("check_technology: модель не ответила (%s) — не проверяю", exc)
        return []
    by_name = {item.name.lower(): label for label, item in labels.items()}
    flagged: dict[str, str] = {}
    for found in answer.not_technology:
        kind = found.kind.strip().lower()
        if found.is_emerging_technology or kind not in _REASONS:
            continue
        label = found.id.strip()
        label = label if label in labels else by_name.get(label.lower())
        if label is not None:
            flagged[label] = kind
    if len(flagged) > len(checked) * MAX_SHARE:
        log.warning(
            "check_technology: модель отметила %d из %d — не верю, никого не отсеиваю", len(flagged), len(checked)
        )
        return []
    decisions = [
        FilterDecision(
            candidate_id=labels[label].candidate_id,
            name=labels[label].name,
            excluded=True,
            reason_code=_REASONS[kind][0],
            reason_text=_REASONS[kind][1],
        )
        for label, kind in flagged.items()
    ]
    log.info("check_technology: из %d верхних кандидатов не технология %d", len(checked), len(decisions))
    return decisions

"""Проверка, что кандидат — технология, а не понятие науки, не зрелый стандарт и не продукт.

Прогоны 27–28.09 на GigaChat 2 Pro и Max: много мусора в топ-15 — понятия и методы математики, физики,
теории управления из статей («шары спектральной нормы», «управление строем», «базисы Грёбнера»), давно
массовые стандарты (AES-256, JPEG2000) и названия продуктов. Такие кандидаты уходят в отсеянные с причиной,
как требует ТЗ. Модель размечает каждого из верхних 45 (на вопрос «найди» она 28.09 в пяти областях из
шести вернула пустой список). Методы машинного обучения не отсеиваем: датасет считает технологией многие
приёмы ИИ, и разметка «метод» задевала технологии датасета. Повтор на 24 сохранённых списках (GigaChat 2 Max):
отсеяно 38, технологий датасета и настоящих слабых сигналов из ручной разметки среди них — 0.
"""

from typing import Literal

from pydantic import BaseModel

from src.common.logs import get_logger
from src.common.schemas import FilterDecision, ScoredCandidate
from src.llm.client import LLMClient, LLMError
from src.llm.prompt_loader import render
from src.pipeline.topic import CHECKED

log = get_logger(__name__)

# Разметка каждого из 45 кандидатов длиннее ответа «найди»: с запасом, обрезанный ответ не пройдёт схему.
MAX_TOKENS = 2500

# Больше этой доли — ответу не верим. Предел выше, чем у проверки темы: в слабых областях методов
# и понятий бывает до трети верхних кандидатов (Max 28.09, индустриальный ИИ: 10 из 15 карточек — мусор).
MAX_SHARE = 1 / 2
_REASONS: dict[str, tuple[Literal["mature", "noise"], str]] = {
    # Методы машинного обучения не отсеиваем: датасет считает технологией многие приёмы ИИ, которые
    # стали продуктами, и 28.09 разметка «метод» задела технологию датасета. Отсеиваем только
    # понятия математики, физики и теории управления — 28.09 это был почти весь мусор в индустриальном ИИ.
    "science": ("noise", "Не технология: понятие или метод математики, физики, теории управления"),
    "standard": ("mature", "Зрелый стандарт или массовый продукт"),
    "product": ("noise", "Название конкретного продукта или модели, а не технология"),
}


class _Item(BaseModel):
    id: str
    kind: str = "technology"
    # Самопроверка: не зарождающаяся ли это всё-таки технология. Нет ответа — верим разметке вида:
    # 28.09 GigaChat 2 Max разметил 12 методов верно, но поле у них не заполнил, и при «нет ответа —
    # технология» фильтр не отсеял никого.
    is_emerging_technology: bool | None = None


class _Answer(BaseModel):
    # Модель размечает каждого кандидата, а не ищет «не технологии». Прогон 28.09 (GigaChat 2 Max):
    # на вопрос «найди» она в пяти областях из шести вернула пустой список, хотя в верхних 45 стояли
    # явные методы из статей; разметка всех по одному закрывает эту лазейку.
    items: list[_Item]


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
    for found in answer.items:
        kind = found.kind.strip().lower()
        if found.is_emerging_technology is not False or kind not in _REASONS:
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

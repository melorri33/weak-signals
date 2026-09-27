"""Проверка, что кандидат относится к области запроса.

Ночной прогон 27.09 (GigaChat 2 Lite, 6 областей датасета): в топ-15 попадали технологии из чужих
областей — «transthyretin amyloid cardiomyopathy» по запросу о защите ИИ, «sustainable sweeteners» —
об индустриальном ИИ. Их приносят случайные документы: научный поиск по фразе запроса отвечает
и статьями из медицины и химии, а выделение кандидатов о запросе не знает. Такие кандидаты уходят
в отсеянные с причиной, как требует ТЗ.
"""

from pydantic import BaseModel

from src.common.logs import get_logger
from src.common.schemas import FilterDecision, ScoredCandidate
from src.llm.client import LLMClient, LLMError
from src.llm.prompt_loader import render

log = get_logger(__name__)

# Сколько верхних кандидатов проверяем: топ-15 с запасом на отсеянных здесь и на тех, у кого
# не соберётся карточка. Ниже по списку чужая тема до выдачи не доходит, проверять её незачем.
CHECKED = 45
# С запасом на случай, когда модель перечисляет много: обрезанный ответ не проходит схему.
MAX_TOKENS = 1500
# Больше этой доли в чужие — ответу не верим. 27.09 на трёхуровневой оценке GigaChat 2 Lite записал
# в чужие 24 из 45 по запросу о защите ИИ, включая атаки на модели. Отсеять треть верхних кандидатов
# по теме — это уже не случайные документы, а непонимание запроса.
MAX_SHARE = 1 / 3


class _OffTopic(BaseModel):
    id: str
    area: str = ""
    # Самопроверка модели: можно ли применить технологию в области запроса. Прогоны 27.09 (Qwen3 14B
    # и GigaChat 2 Pro): без неё фильтр записывал в чужие технологии ИИ, навигации и управления по
    # запросам о финтехе и робототехнике с областью «искусственный интеллект», хотя промпт просил
    # так не делать, — и выбросил две технологии датасета. Нет ответа — считаем применимой.
    applies_to_query: bool = True


class _Answer(BaseModel):
    off_topic: list[_OffTopic]


async def off_topic(query: str, scored: list[ScoredCandidate], client: LLMClient | None = None) -> list[FilterDecision]:
    """Решения об отсеве для верхних кандидатов из чужой области. Модель не ответила — никого не отсеиваем."""
    checked = scored[:CHECKED]
    if not checked:
        return []
    labels = {f"t{i}": item for i, item in enumerate(checked, start=1)}
    prompt = render("check_topic", query=query, candidates="\n".join(f"- {k}: {v.name}" for k, v in labels.items()))
    try:
        client = client or LLMClient.from_settings()
        answer = await client.ask_json(step="check_topic", prompt=prompt, schema=_Answer, max_tokens=MAX_TOKENS)
    except LLMError as exc:
        log.warning("check_topic: модель не ответила (%s) — тему не проверяю", exc)
        return []
    decisions = []
    # GigaChat иногда пишет вместо id само название — принимаем и так.
    by_name = {item.name.lower(): label for label, item in labels.items()}
    flagged = {}
    for found in answer.off_topic:
        if found.applies_to_query:
            continue  # модель сама признала, что технология применима в области запроса
        label = found.id.strip()
        label = label if label in labels else by_name.get(label.lower())
        if label is not None:
            flagged[label] = found.area.strip()
    if len(flagged) > len(checked) * MAX_SHARE:
        log.warning(
            "check_topic: модель записала в чужие %d из %d — не верю, никого не отсеиваю", len(flagged), len(checked)
        )
        return []
    for label, area in flagged.items():
        item = labels[label]
        decisions.append(
            FilterDecision(
                candidate_id=item.candidate_id,
                name=item.name,
                excluded=True,
                reason_code="noise",
                reason_text=f"Не по теме запроса: {area}" if area else "Не по теме запроса",
            )
        )
    log.info("check_topic: из %d верхних кандидатов не по теме %d", len(checked), len(decisions))
    return decisions

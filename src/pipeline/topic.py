"""Проверка, что кандидат относится к области запроса.

Ночной прогон 27.09 (GigaChat 2 Lite, 6 областей датасета): в топ-15 попадали технологии из чужих
областей — «transthyretin amyloid cardiomyopathy» по запросу о защите ИИ, «sustainable sweeteners» —
об индустриальном ИИ. Их приносят случайные документы: научный поиск по фразе запроса отвечает
и статьями из медицины и химии, а выделение кандидатов о запросе не знает. Такие кандидаты уходят
в отсеянные с причиной, как требует ТЗ.

29.09 (GigaChat 2 Max): из 15 мусорных карточек лучшего прогона 8 были чужой темой — кубиты, медицинские
цифровые двойники, синтез речи. Проверка списком («найди чужих») была непоследовательна: кубиты по финтеху
отсеяла, а в индустриальном ИИ и инфраструктуре ИИ оставила. Теперь модель размечает каждого кандидата,
и спрашиваем её дважды, в прямом и обратном порядке списка: одна и та же разметка по защите ИИ давала
то 0, то 15 чужих, а отсеиваем только тех, кого модель оба раза записала в чужие.
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
# Разметка каждого из 45 кандидатов с областью длиннее ответа «найди»: обрезанный ответ не пройдёт схему.
MAX_TOKENS = 3000
# Больше этой доли в чужие — ответу не верим. 27.09 на трёхуровневой оценке GigaChat 2 Lite записал
# в чужие 24 из 45 по запросу о защите ИИ, включая атаки на модели. Отсеять треть верхних кандидатов
# по теме — это уже не случайные документы, а непонимание запроса.
MAX_SHARE = 1 / 3


class _Item(BaseModel):
    id: str
    area: str = ""
    fit: str = "core"
    # Самопроверка модели: можно ли применить технологию в области запроса. Прогоны 27.09 (Qwen3 14B
    # и GigaChat 2 Pro): без неё фильтр записывал в чужие технологии ИИ, навигации и управления по
    # запросам о финтехе и робототехнике с областью «искусственный интеллект», хотя промпт просил
    # так не делать, — и выбросил две технологии датасета. Нет ответа — считаем применимой.
    applies_to_query: bool | None = None


class _Answer(BaseModel):
    items: list[_Item]


async def off_topic(query: str, scored: list[ScoredCandidate], client: LLMClient | None = None) -> list[FilterDecision]:
    """Решения об отсеве для верхних кандидатов из чужой области. Модель не ответила — никого не отсеиваем."""
    checked = scored[:CHECKED]
    if not checked:
        return []
    labels = {f"t{i}": item for i, item in enumerate(checked, start=1)}
    try:
        client = client or LLMClient.from_settings()
        first = await _aliens(client, query, labels, list(labels))
        second = await _aliens(client, query, labels, list(reversed(labels)))
    except LLMError as exc:
        log.warning("check_topic: модель не ответила (%s) — тему не проверяю", exc)
        return []
    flagged = {label: area or second[label] for label, area in first.items() if label in second}
    if len(flagged) > len(checked) * MAX_SHARE:
        log.warning(
            "check_topic: модель записала в чужие %d из %d — не верю, никого не отсеиваю", len(flagged), len(checked)
        )
        return []
    decisions = [
        FilterDecision(
            candidate_id=labels[label].candidate_id,
            name=labels[label].name,
            excluded=True,
            reason_code="noise",
            reason_text=f"Не по теме запроса: {area}" if area else "Не по теме запроса",
        )
        for label, area in flagged.items()
    ]
    log.info(
        "check_topic: из %d верхних кандидатов не по теме %d (первый ответ — %d, второй — %d)",
        len(checked),
        len(decisions),
        len(first),
        len(second),
    )
    return decisions


async def _aliens(
    client: LLMClient, query: str, labels: dict[str, ScoredCandidate], order: list[str]
) -> dict[str, str]:
    """Один опрос модели: кого она записала в чужую область (id → область). Порядок списка задаёт `order`.

    Второй опрос идёт в обратном порядке: это другой запрос, поэтому кэш ответов его не подменит,
    а место в списке не влияет на решение.
    """
    prompt = render("check_topic", query=query, candidates="\n".join(f"- {k}: {labels[k].name}" for k in order))
    answer = await client.ask_json(step="check_topic", prompt=prompt, schema=_Answer, max_tokens=MAX_TOKENS)
    # GigaChat иногда пишет вместо id само название — принимаем и так.
    by_name = {item.name.lower(): label for label, item in labels.items()}
    found = {}
    for item in answer.items:
        if item.fit.strip().lower() != "other" or item.applies_to_query is not False:
            continue  # своя, соседняя или модель сама признала, что технология применима в области запроса
        label = item.id.strip()
        label = label if label in labels else by_name.get(label.lower())
        if label is not None:
            found[label] = item.area.strip()
    return found

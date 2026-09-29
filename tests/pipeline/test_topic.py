"""Проверка темы: чужие области уходят в отсеянные с причиной, при сомнении в ответе — никто."""

from pydantic import BaseModel

from src.common.schemas import ScoredCandidate
from src.llm.client import LLMError
from src.pipeline.topic import CHECKED, off_topic

QUERY = "перспективные технологии в агротехе"


class _FakeClient:
    """Отвечает по очереди ответами из списка; последний повторяется."""

    def __init__(self, *answers: dict | None):
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def ask_json(self, step: str, prompt: str, schema: type[BaseModel], **kwargs) -> BaseModel:
        self.prompts.append(prompt)
        answer = self.answers[min(len(self.prompts), len(self.answers)) - 1]
        if answer is None:
            raise LLMError("модель не ответила")
        return schema.model_validate(answer)


def _scored(*names: str) -> list[ScoredCandidate]:
    return [ScoredCandidate(candidate_id=f"c{i}", name=n, score=0.9 - i / 100) for i, n in enumerate(names)]


def _items(*rows: tuple) -> dict:
    """Строки (id, fit, applies_to_query[, area]); applies_to_query=None — поле не заполнено."""
    out = []
    for label, fit, applies, *area in rows:
        row = {"id": label, "fit": fit, "area": area[0] if area else ""}
        if applies is not None:
            row["applies_to_query"] = applies
        out.append(row)
    return {"items": out}


SCORED = _scored("soil moisture sensing", "drug screening organoids", "weed spraying drone", "crop yield forecasting")


async def test_alien_area_goes_to_excluded_with_reason():
    client = _FakeClient(
        _items(("t1", "core", True), ("t2", "other", False, "медицина"), ("t3", "core", True), ("t4", "core", True))
    )

    decisions = await off_topic(QUERY, SCORED, client=client)

    assert [(d.candidate_id, d.excluded, d.reason_code) for d in decisions] == [("c1", True, "noise")]
    assert decisions[0].reason_text == "Не по теме запроса: медицина"
    assert QUERY in client.prompts[0] and "t4: crop yield forecasting" in client.prompts[0]


async def test_second_ask_goes_in_reverse_order_and_only_both_times_aliens_are_dropped():
    """29.09 одна и та же разметка по защите ИИ давала то 0, то 15 чужих — верим только совпадению."""
    client = _FakeClient(
        _items(("t2", "other", False, "медицина"), ("t3", "other", False, "дроны")),
        _items(("t2", "other", False, "медицина"), ("t4", "other", False, "прогнозы")),
    )

    decisions = await off_topic(QUERY, SCORED, client=client)

    assert [d.candidate_id for d in decisions] == ["c1"]
    first, second = client.prompts
    assert first.index("t1: soil") < first.index("t4: crop")
    assert second.index("t4: crop") < second.index("t1: soil")


async def test_name_instead_of_id_is_accepted_and_unknown_ids_are_ignored():
    client = _FakeClient(_items(("Drug Screening Organoids", "other", False), ("t99", "other", False)))

    decisions = await off_topic(QUERY, SCORED, client=client)

    assert [d.candidate_id for d in decisions] == ["c1"]


async def test_neighbour_discipline_and_self_check_keep_the_technology():
    """27.09 фильтр записывал технологии ИИ в чужие по запросам о финтехе и роботах; самопроверка их возвращает."""
    client = _FakeClient(
        _items(
            ("t1", "applied", False, "искусственный интеллект"),
            ("t2", "other", False, "медицина"),
            ("t3", "other", True, "навигация"),
            ("t4", "other", None, "прогнозы"),
        )
    )

    decisions = await off_topic(QUERY, SCORED, client=client)

    assert [d.candidate_id for d in decisions] == ["c1"]
    assert "applies_to_query" in client.prompts[0]


async def test_model_that_rejects_too_much_is_not_trusted():
    scored = _scored("soil moisture sensing", "drug screening organoids", "weed spraying drone")
    client = _FakeClient(_items(("t1", "other", False), ("t2", "other", False)))

    assert await off_topic(QUERY, scored, client=client) == []


async def test_no_answer_drops_nobody():
    assert await off_topic(QUERY, SCORED, client=_FakeClient(None)) == []


async def test_no_second_answer_drops_nobody():
    client = _FakeClient(_items(("t2", "other", False, "медицина")), None)

    assert await off_topic(QUERY, SCORED, client=client) == []


async def test_only_the_top_of_the_list_is_checked():
    scored = _scored(*[f"technology {i}" for i in range(CHECKED + 10)])
    client = _FakeClient({"items": []})

    await off_topic(QUERY, scored, client=client)

    assert f"technology {CHECKED - 1}" in client.prompts[0]
    assert f"technology {CHECKED}\n" not in client.prompts[0] + "\n"

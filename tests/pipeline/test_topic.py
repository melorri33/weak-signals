"""Проверка темы: чужие области уходят в отсеянные с причиной, при сомнении в ответе — никто."""

from pydantic import BaseModel

from src.common.schemas import ScoredCandidate
from src.llm.client import LLMError
from src.pipeline.topic import CHECKED, off_topic

QUERY = "перспективные технологии в агротехе"


class _FakeClient:
    def __init__(self, answer: dict | None):
        self.answer = answer
        self.prompts: list[str] = []

    async def ask_json(self, step: str, prompt: str, schema: type[BaseModel], **kwargs) -> BaseModel:
        self.prompts.append(prompt)
        if self.answer is None:
            raise LLMError("модель не ответила")
        return schema.model_validate(self.answer)


def _scored(*names: str) -> list[ScoredCandidate]:
    return [ScoredCandidate(candidate_id=f"c{i}", name=n, score=0.9 - i / 100) for i, n in enumerate(names)]


async def test_alien_area_goes_to_excluded_with_reason():
    scored = _scored(
        "soil moisture sensing", "drug screening organoids", "weed spraying drone", "crop yield forecasting"
    )
    client = _FakeClient({"off_topic": [{"id": "t2", "area": "медицина"}]})

    decisions = await off_topic(QUERY, scored, client=client)

    assert [(d.candidate_id, d.excluded, d.reason_code) for d in decisions] == [("c1", True, "noise")]
    assert decisions[0].reason_text == "Не по теме запроса: медицина"
    assert QUERY in client.prompts[0] and "t4: crop yield forecasting" in client.prompts[0]


async def test_name_instead_of_id_is_accepted_and_unknown_ids_are_ignored():
    scored = _scored(
        "soil moisture sensing", "drug screening organoids", "weed spraying drone", "crop yield forecasting"
    )
    client = _FakeClient({"off_topic": [{"id": "Drug Screening Organoids"}, {"id": "t99"}]})

    decisions = await off_topic(QUERY, scored, client=client)

    assert [d.candidate_id for d in decisions] == ["c1"]


async def test_model_that_rejects_too_much_is_not_trusted():
    scored = _scored("soil moisture sensing", "drug screening organoids", "weed spraying drone")
    client = _FakeClient({"off_topic": [{"id": "t1"}, {"id": "t2"}]})

    assert await off_topic(QUERY, scored, client=client) == []


async def test_no_answer_drops_nobody():
    assert await off_topic(QUERY, _scored("soil moisture sensing"), client=_FakeClient(None)) == []


async def test_only_the_top_of_the_list_is_checked():
    scored = _scored(*[f"technology {i}" for i in range(CHECKED + 10)])
    client = _FakeClient({"off_topic": []})

    await off_topic(QUERY, scored, client=client)

    assert f"technology {CHECKED - 1}" in client.prompts[0]
    assert f"technology {CHECKED}\n" not in client.prompts[0] + "\n"

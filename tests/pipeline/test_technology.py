"""Проверка «это технология?»: методы из статей и зрелые стандарты уходят в отсеянные с причиной."""

from pydantic import BaseModel

from src.common.schemas import ScoredCandidate
from src.llm.client import LLMError
from src.pipeline.technology import not_technology

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


SCORED = _scored("soil moisture sensing", "finite element method", "gps tractor guidance", "weed spraying drone")


async def test_method_and_standard_go_to_excluded_with_reason():
    client = _FakeClient(
        {
            "not_technology": [
                {"id": "t2", "kind": "method", "is_emerging_technology": False},
                {"id": "t3", "kind": "standard", "is_emerging_technology": False},
            ]
        }
    )

    decisions = await not_technology(QUERY, SCORED, client=client)

    assert [(d.candidate_id, d.reason_code) for d in decisions] == [("c1", "noise"), ("c2", "mature")]
    assert decisions[0].reason_text.startswith("Не технология")
    assert "t4: weed spraying drone" in client.prompts[0]


async def test_self_check_and_unknown_kind_keep_the_candidate():
    client = _FakeClient(
        {
            "not_technology": [
                {"id": "t2", "kind": "method", "is_emerging_technology": True},
                {"id": "t3", "kind": "product", "is_emerging_technology": False},
                {"id": "t4", "kind": "method"},
            ]
        }
    )

    assert await not_technology(QUERY, SCORED, client=client) == []


async def test_model_that_flags_more_than_half_is_not_trusted():
    flags = [{"id": f"t{i}", "kind": "method", "is_emerging_technology": False} for i in (1, 2, 3)]

    assert await not_technology(QUERY, SCORED, client=_FakeClient({"not_technology": flags})) == []


async def test_no_answer_drops_nobody():
    assert await not_technology(QUERY, SCORED, client=_FakeClient(None)) == []

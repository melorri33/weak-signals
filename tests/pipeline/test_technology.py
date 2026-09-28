"""Проверка «это технология?»: понятия науки, стандарты и продукты — в отсеянные с причиной, методы ИИ — нет."""

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


SCORED = _scored(
    "soil moisture sensing",
    "finite element method",
    "gps tractor guidance",
    "tractor model x9",
    "weed spraying drone",
    "crop yield forecasting",
    "plant disease detection model",
)


def _items(*rows: tuple[str, str, bool | None]) -> dict:
    out = []
    for label, kind, emerging in rows:
        row = {"id": label, "kind": kind}
        if emerging is not None:
            row["is_emerging_technology"] = emerging
        out.append(row)
    return {"items": out}


async def test_science_standard_and_product_go_to_excluded_with_reason():
    client = _FakeClient(
        _items(
            ("t1", "technology", True), ("t2", "science", False), ("t3", "standard", False), ("t4", "product", False)
        )
    )

    decisions = await not_technology(QUERY, SCORED, client=client)

    assert [(d.candidate_id, d.reason_code) for d in decisions] == [("c1", "noise"), ("c2", "mature"), ("c3", "noise")]
    assert decisions[0].reason_text.startswith("Не технология")
    assert "t5: weed spraying drone" in client.prompts[0]


async def test_ml_method_self_check_and_missing_answer_keep_the_candidate():
    """Методы ИИ датасет считает технологиями; «всё же технология» и пропущенное поле — тоже оставляем."""
    client = _FakeClient(
        _items(("t7", "ml_method", False), ("t2", "science", True), ("t4", "product", None), ("t6", "unknown", False))
    )

    assert await not_technology(QUERY, SCORED, client=client) == []


async def test_model_that_flags_more_than_half_is_not_trusted():
    flags = _items(*[(f"t{i}", "science", False) for i in (1, 2, 3, 4)])

    assert await not_technology(QUERY, SCORED, client=_FakeClient(flags)) == []


async def test_no_answer_drops_nobody():
    assert await not_technology(QUERY, SCORED, client=_FakeClient(None)) == []

"""Проверка живой связки с Ollama. Требует запущенного Ollama и скачанной модели.

Запуск: pytest -m network tests/llm/test_client.py
"""

import pytest
from pydantic import BaseModel

from src.llm.client import LLMClient

pytestmark = pytest.mark.network


class _Answer(BaseModel):
    city: str
    year: int


async def test_json_answer_matches_schema():
    client = LLMClient.from_settings()
    if not await client.is_available():
        pytest.skip("Ollama не запущена или модель не скачана")

    answer = await client.ask_json(
        step="test",
        prompt="В каком городе и в каком году прошли первые Олимпийские игры современности? Ответь JSON.",
        schema=_Answer,
    )

    assert answer.year == 1896
    assert "Афин" in answer.city or "Athen" in answer.city

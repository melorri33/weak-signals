"""Проверка живой связки с моделью: локальной через Ollama и облачной через YandexGPT.

Запуск: pytest -m network tests/llm/test_client.py
Облачный тест пропускается, пока в .env нет ключей.
"""

import pytest
from pydantic import BaseModel

from src.common.config import get_settings
from src.llm.client import LLMClient
from src.llm.providers import make_backend

pytestmark = pytest.mark.network

CLOUD_MODEL = "yandexgpt-5-lite"


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

    # Проверяем механизм: ответ разобран в модель pydantic, типы правильные.
    # Написание города у модели своё («Афины», «Атена», «Athens») — на это не опираемся.
    assert answer.year == 1896
    assert answer.city.strip()


async def test_cloud_json_answer_matches_schema():
    """То же самое на облачной модели. Пропускается, пока в .env нет ключей YandexGPT.

    Тест для того, у кого есть доступ к облаку: он же покажет время ответа
    (`pytest -m network -k cloud --durations=0`).
    """
    settings = get_settings()
    if not settings.yandex_api_key or not settings.yandex_folder_id:
        pytest.skip("нет YANDEX_API_KEY / YANDEX_FOLDER_ID в .env")

    client = LLMClient(
        model=CLOUD_MODEL,
        provider="yandexgpt",
        backend=make_backend("yandexgpt", CLOUD_MODEL),
    )
    answer = await client.ask_json(
        step="test",
        prompt="В каком городе и в каком году прошли первые Олимпийские игры современности? Ответь JSON.",
        schema=_Answer,
    )
    assert answer.year == 1896
    assert answer.city.strip()

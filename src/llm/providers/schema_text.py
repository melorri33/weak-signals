"""JSON-схема текстом в системном сообщении — для облаков, где структурированный ответ не работает.

GigaChat `response_format` принимает, но не соблюдает. YandexGPT Lite в режиме `json_schema` ломает
ключи на вложенных списках: `"document_ids»: [": "d1"` — на ночном прогоне 26.09 ни одна из 63 пачек
выделения кандидатов не прошла, и конвейер резал названия из заголовков. С той же схемой текстом —
настоящие технологии. Проверку схемой и повтор при ошибке делает LLMClient.ask_json.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"\s*```(?:json)?\s*(.*?)\s*```\s*", re.DOTALL)


def with_schema(messages: list[dict[str, str]], json_schema: dict) -> list[dict[str, str]]:
    """Добавить JSON-схему в системное сообщение (или создать его)."""
    rule = "Ответ — только один JSON-объект строго по этой JSON-схеме, без пояснений и без ```:\n" + json.dumps(
        json_schema, ensure_ascii=False
    )
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": f"{messages[0]['content']}\n\n{rule}"}, *messages[1:]]
    return [{"role": "system", "content": rule}, *messages]


def strip_fence(content: str) -> str:
    """Модель иногда оборачивает JSON в ```json … ``` — схема такое не пропустит."""
    match = _FENCE_RE.fullmatch(content)
    return match.group(1) if match else content

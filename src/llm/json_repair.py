"""Починка частых огрехов JSON в ответе модели — до того, как тратить повторный вызов.

Облачные модели без строгого структурированного вывода (GigaChat, YandexGPT со схемой в промпте)
иногда отвечают почти правильным JSON. Прогон 26.09 на GigaChat 2 Lite: из 15 карточек области две
остались без текста, и в обоих случаях ответ был верным по содержанию, но:
- с лишней запятой перед `]` или `}`;
- на повторной попытке модель сначала переписывала саму схему (`{"$defs": …} }`, с лишней скобкой),
  а следом — ответ.

Чиним только эти два случая и только когда ответ не разбирается как есть. Обрезанный по длине ответ
не чиним: достраивать недописанный текст — значит выдумывать его.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Запятая, за которой (через пробелы) закрывается объект или список. Внутри строк такое почти не
# встречается, а починка применяется, только если ответ как есть не разобрался.
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
# Ключи, по которым видно, что модель переписала JSON-схему, а не ответила по ней.
_SCHEMA_KEYS = {"$defs", "$schema"}


def repair_json(content: str) -> str:
    """Ответ модели → JSON, который разберётся. Не вышло — ответ без лишних запятых, ошибку покажет схема."""
    if _parses(content):
        return content
    fixed = _TRAILING_COMMA_RE.sub(r"\1", content)
    answers = [obj for obj in _objects(fixed) if _looks_like_answer(obj)]
    if answers:
        return json.dumps(answers[-1], ensure_ascii=False)
    return fixed


def _parses(content: str) -> bool:
    try:
        json.loads(content)
    except ValueError:
        return False
    return True


def _objects(content: str) -> list[Any]:
    """JSON-значения, записанные в строке одно за другим; на первом неразборном месте — стоп.

    Между значениями пропускаем пробелы и одиночные `}`, `]`, `,`: их модель оставляет после схемы.
    """
    decoder = json.JSONDecoder()
    found: list[Any] = []
    pos = 0
    while (pos := _skip_between(content, pos)) < len(content):
        try:
            value, pos = decoder.raw_decode(content, pos)
        except ValueError:
            break
        found.append(value)
    return found


def _skip_between(content: str, pos: int) -> int:
    while pos < len(content) and (content[pos].isspace() or content[pos] in "}],"):
        pos += 1
    return pos


def _looks_like_answer(value: Any) -> bool:
    """Ответ по схеме pydantic — всегда объект, и это не сама схема."""
    if not isinstance(value, dict):
        return False
    return not (_SCHEMA_KEYS & value.keys()) and not (value.get("type") == "object" and "properties" in value)

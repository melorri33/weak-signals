"""Починка частых огрехов JSON в ответе модели — до того, как тратить повторный вызов.

Облачные модели без строгого структурированного вывода (GigaChat, YandexGPT со схемой в промпте)
иногда отвечают почти правильным JSON. Прогон 26.09 на GigaChat 2 Lite: из 15 карточек области две
остались без текста, и в обоих случаях ответ был верным по содержанию, но:
- с лишней запятой перед `]` или `}`;
- на повторной попытке модель сначала переписывала саму схему (`{"$defs": …} }`, с лишней скобкой),
  а следом — ответ.

Прогон 27.09 на GigaChat 2 Pro: 33 карточки из 90 без текста, 99 ответов из 153 не разбирались.
Текст в них был, ломалась разметка вокруг него:
- лишняя или перепутанная скобка в конце (`"}]]}`, `"}])}`);
- кавычки внутри текста без экранирования;
- удвоенная кавычка в конце значения;
- незакрытый объект перед следующим элементом списка.

Эти огрехи чинит `_balance`: только разметку, текст не трогает. Из тех 99 ответов проходят схему 97.

Чиним только когда ответ не разбирается как есть. Обрезанный по длине ответ не чиним: достраивать
недописанный текст — значит выдумывать его. Поэтому разметку правим, только если ответ закончен:
последний знак — закрывающая скобка.
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

# Закрывающие знаки, которыми кончается законченный ответ (круглую модель ставит вместо фигурной).
_CLOSERS = ("}", "]", ")")
_PAIR = {"{": "}", "[": "]"}
# Что может стоять после закрывающей кавычки строки. Кавычка, за которой идёт другое, — внутри текста.
_AFTER_STRING = ",:}])"


def repair_json(content: str) -> str:
    """Ответ модели → JSON, который разберётся. Не вышло — ответ без лишних запятых, ошибку покажет схема."""
    if _parses(content):
        return content
    fixed = _TRAILING_COMMA_RE.sub(r"\1", content)
    answers = [obj for obj in _objects(fixed) if _looks_like_answer(obj)]
    if answers:
        return json.dumps(answers[-1], ensure_ascii=False)
    if not fixed.rstrip().endswith(_CLOSERS):
        return fixed  # ответ не закончен — вероятно, обрезан по длине
    balanced = _balance(fixed.strip())
    try:
        value = json.loads(balanced)
    except ValueError:
        return fixed
    return balanced if _looks_like_answer(value) else fixed


def _balance(content: str) -> str:
    """Выправить разметку вокруг текста: скобки, кавычки внутри строк, незакрытые элементы."""
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    i = 0
    while i < len(content):
        ch = content[i]
        if in_string:
            i, in_string = _string_char(content, i, out)
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch in "{[":
            last = "".join(out).rstrip()
            if ch == "{" and stack and stack[-1] == "{" and last.endswith(","):
                # новый объект там, где ждали ключ: модель не закрыла предыдущий элемент списка
                out = [last[:-1], "},"]
                stack.pop()
            stack.append(ch)
            out.append(ch)
        elif ch in "}]" and ch in (_PAIR[opened] for opened in stack):
            while _PAIR[stack[-1]] != ch:
                out.append(_PAIR[stack.pop()])  # забытые внутренние скобки закрываем
            stack.pop()
            out.append(ch)
            if not stack:
                break  # корневой объект закрыт — дальше мусор
        elif ch not in "}])":
            out.append(ch)  # лишние и круглые скобки вне строк пропускаем
        i += 1
    if in_string:
        out = ["".join(out).rstrip("}]) \t\r\n"), '"']  # незакрытая строка: скобочный мусор — не текст
    out.extend(_PAIR[opened] for opened in reversed(stack))
    return _TRAILING_COMMA_RE.sub(r"\1", "".join(out))


def _string_char(content: str, i: int, out: list[str]) -> tuple[int, bool]:
    """Один знак внутри строки → (следующая позиция, остаёмся ли в строке)."""
    ch = content[i]
    if ch == "\\":
        out.append(content[i : i + 2])
        return i + 2, True
    if ch == '"' and content[i + 1 : i + 2] == '"' and _next_char(content, i + 2) in ("", *_AFTER_STRING):
        out.append(ch)  # удвоенная кавычка в конце значения: закрываем строку, лишнюю пропускаем
        return i + 2, False
    if ch == '"':
        if _next_char(content, i + 1) in ("", *_AFTER_STRING):
            out.append(ch)
            return i + 1, False
        out.append('\\"')  # кавычка внутри текста
        return i + 1, True
    out.append({"\n": "\\n", "\r": " ", "\t": " "}.get(ch, ch))
    return i + 1, True


def _next_char(content: str, pos: int) -> str:
    while pos < len(content) and content[pos].isspace():
        pos += 1
    return content[pos] if pos < len(content) else ""


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

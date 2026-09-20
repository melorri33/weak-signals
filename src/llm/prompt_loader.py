"""Загрузка промптов из src/llm/prompts/*.md.

Промпты держим отдельными файлами, а не в коде: их правит Продукт вместе с Интеграцией,
и так видно историю правок в git.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache
def load_prompt(name: str) -> str:
    """Текст промпта по имени файла без расширения, например load_prompt('expand_query')."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Нет промпта {path}")
    return path.read_text(encoding="utf-8").strip()


def render(prompt_name: str, **values: str) -> str:
    """Подставить значения в промпт: в файле места подстановки помечены как {query}, {documents}.

    Первый параметр назван prompt_name, а не name: у промпта карточки есть подстановка {name},
    и одноимённый параметр функции конфликтовал бы с ней.
    """
    return load_prompt(prompt_name).format(**values)

"""Фразы свежести: поиск новостей о молодых компаниях области запроса.

Фразы, которые придумывает модель, — это темы из её памяти, то есть устоявшееся: для робототехники
«soft robotic grippers», «robotic SLAM algorithms», для защиты ИИ «homomorphic encryption»,
«differential privacy». Поиск по ним тянет литературу пятилетней давности. А ранние технологии живут
в свежих новостях о стартапах: треть описаний в датасете организаторов — про свежие раунды.

Замер 27.09, только новостные ленты, шесть областей: по 18 фразам модели — 4457 документов и 45
технологий датасета (1.0 на сотню документов), по пяти фразам свежести — 1420 документов и 32
технологии (2.3 на сотню); медиана даты документов — январь–июль 2026 против августа–сентября.
Модель читает несколько сотен документов из тысяч, поэтому решает плотность, а не общее число.

Фразы свежести ищутся только в новостях: научный поиск по «robotics startup raises» приносит мусор.
Общие между модулями (расширение запроса их строит, сбор по ним решает, куда искать), поэтому здесь.
"""

from __future__ import annotations

FRESH_TEMPLATES = (
    "{area} startup raises",
    "{area} startup funding round",
    "{area} startup emerges from stealth",
    "{area} startup launches",
    "{area} seed round",
    # 28.09: не только раунды — запуски, разработки лабораторий и пилоты; проверяются полным прогоном.
    "{area} startup unveils",
    "{area} researchers develop",
    "{area} pilot deployment",
)

_FRESH_TAILS = tuple(template.split("{area} ", 1)[1] for template in FRESH_TEMPLATES)


def fresh_phrases(area: str) -> list[str]:
    """Фразы свежести для области («robotics» → «robotics startup raises», …). Пустая область — пусто."""
    area = " ".join(area.split())
    if not area:
        return []
    return [template.format(area=area) for template in FRESH_TEMPLATES]


def is_fresh_phrase(phrase: str | None) -> bool:
    """Фраза свежести ищется только в новостях, а её документы модель читает первыми."""
    return bool(phrase) and phrase.strip().lower().endswith(_FRESH_TAILS)

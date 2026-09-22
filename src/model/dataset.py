"""Обучающая выборка этапа 1: сигналы из датасета организаторов + наши отрицательные примеры.

Датасет организаторов (xlsx) содержит только положительный класс — 100 слабых сигналов.
Отрицательные (зрелые, массовые, но растущие, хайп, шум — в тех же областях) размечены нами: `training/negatives.csv`.

Виды `product`, `too_broad`, `not_a_technology` добавлены 22.09 по разметке настоящей выдачи
конвейера: названия продуктов, целые направления вместо технологий, процессы и роли. Из 74
размеченных в обучение взяты только 30 — те, у кого больше 200 публикаций по точной фразе.
Причина в том, что признаки модели описывают только динамику публикаций: у продукта с тремя
работами профиль неотличим от слабого сигнала, и такой пример учит топить то, что мы ищем.
Замер: со всеми 74 accuracy падает 0.836 -> 0.822 и ломается порядок «растущая выше плоской»
(tests/model/test_score.py), с отобранными 30 accuracy растёт до 0.843, PR-AUC 0.714 -> 0.748.
Остальные 44 отсеиваются правилами по названию (src/pipeline/candidates.py), а не моделью.
Английские поисковые термины для сигналов — `data/positive_terms.csv` (производное от датасета, не коммитим).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.common.schemas import Stage

TRAINING_DIR = Path(__file__).parent / "training"
NEGATIVES_PATH = TRAINING_DIR / "negatives.csv"

# Столбцы xlsx организаторов → наши имена.
_SIGNAL_COLUMNS = {
    "№": "id",
    "Технология (слабый сигнал)": "name_ru",
    "Область": "domain",
    "Компании": "companies",
    "Почему это слабый сигнал": "why",
    "Стадия развития": "stage_raw",
    "Тренд упоминаний": "trend_raw",
    "Балл (стадия+тренд)": "expert_score",
    "Источники": "sources_md",
}

# Ключевые слова стадий из столбца «Стадия развития».
_STAGE_PATTERNS: list[tuple[str, Stage]] = [
    (r"концепц|исследован", "research"),
    (r"прототип|poc|пилот", "prototype"),
    (r"раннее внедрение|ранние внедрения|ранняя серия|серийн", "product"),
    (r"массов", "mass"),
]


def normalize_stage(text: str) -> Stage | None:
    """Свести свободный текст стадии к Stage.

    Берётся часть до стрелки «→» (текущая стадия), а в ней — стадия, упомянутая первой:
    «Раннее внедрение (у лидера) / Прототип (у остальных)» → product.
    """
    current = text.split("→")[0].lower()
    found = [(m.start(), stage) for pattern, stage in _STAGE_PATTERNS if (m := re.search(pattern, current))]
    return min(found)[1] if found else None


def load_signals(xlsx_path: Path, terms_path: Path | None = None) -> pd.DataFrame:
    """Загрузить сигналы организаторов. Если есть файл терминов — добавить колонку term_en."""
    df = pd.read_excel(xlsx_path, header=1)
    df = df[list(_SIGNAL_COLUMNS)].rename(columns=_SIGNAL_COLUMNS)
    df = df.dropna(subset=["id", "name_ru"]).astype({"id": int, "expert_score": int})
    df["stage"] = df["stage_raw"].map(normalize_stage)
    df["stage_moving"] = df["stage_raw"].str.contains("→")
    df["source_urls"] = df["sources_md"].fillna("").map(lambda s: re.findall(r"\((https?://[^)\s]+)\)", s))
    if terms_path is not None:
        terms = pd.read_csv(terms_path)
        df = df.merge(terms, on="id", how="left")
    return df.reset_index(drop=True)


def load_negatives(path: Path = NEGATIVES_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_labeled_set(xlsx_path: Path, terms_path: Path) -> pd.DataFrame:
    """Единая таблица для обучения: key, term_en, name_ru, domain, label (1 — слабый сигнал), kind."""
    pos = load_signals(xlsx_path, terms_path)
    missing = pos[pos["term_en"].isna()]["id"].tolist()
    if missing:
        raise ValueError(f"Нет английского термина для сигналов: {missing}")
    pos_part = pd.DataFrame(
        {
            "key": "pos-" + pos["id"].astype(str),
            "term_en": pos["term_en"],
            "name_ru": pos["name_ru"],
            "domain": pos["domain"],
            "label": 1,
            "kind": "weak_signal",
        }
    )
    neg = load_negatives()
    neg_part = pd.DataFrame(
        {
            "key": "neg-" + neg.index.astype(str),
            "term_en": neg["term_en"],
            "name_ru": neg["name_ru"],
            "domain": neg["domain"],
            "label": 0,
            "kind": neg["kind"],
        }
    )
    return pd.concat([pos_part, neg_part], ignore_index=True)

"""Модель названий: похоже ли название кандидата на узкую раннюю технологию.

Зачем вторая модель. CatBoost судит о новизне по публикациям с точной фразой, поэтому необычная
формулировка выглядит новой: у «explanable ai» с опечаткой ноль работ, и модель давала ей 0.92.
Ручная разметка 90 карточек 23.09: первое место в четырёх областях из шести — ошибка
с уверенностью 0.97–0.99, а из 90 мест 25 заняли слишком широкие названия («edge ai inference»),
20 — зрелые методы, 4 — продукты.

Эта модель смотрит на смысл самого названия: эмбеддинг bge-m3 и логистическая регрессия поверх.
Учится на терминах датасета организаторов (слабые сигналы), наших отрицательных примерах,
ручной разметке кандидатов из настоящих прогонов и синтетических примерах из областей вне
датасета (src/model/training). Обучение: `python -m src.model.train_names`.

Итоговая уверенность — смесь: BLEND от модели названий, остальное от CatBoost. Нет артефакта
или sentence-transformers — возвращаем None, и конвейер работает на одном CatBoost.
Эмбеддинги считаются на процессоре: видеопамять занята языковой моделью.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.common.logs import get_logger, model_timer

NAME_MODEL_PATH = Path(__file__).parent / "artifacts" / "name_model.json"
PROVIDER = "local"

log = get_logger(__name__)


@lru_cache(maxsize=1)
def artifact() -> dict | None:
    """Веса модели названий или None, если модель ещё не обучена."""
    if not NAME_MODEL_PATH.exists():
        log.warning(
            "Нет модели названий %s — оценка только по CatBoost. Обучить: python -m src.model.train_names",
            NAME_MODEL_PATH,
        )
        return None
    return json.loads(NAME_MODEL_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _encoder(model_name: str):  # -> SentenceTransformer
    """Сначала с диска: иначе каждая загрузка — десятки запросов к Hugging Face, даже при скачанной модели."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except OSError:
        log.info("Модели %s нет на диске — скачиваю (около 2 ГБ, один раз)", model_name)
        return SentenceTransformer(model_name, device="cpu")


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """Нормированные эмбеддинги названий (общие для обучения и инференса)."""
    return np.asarray(
        _encoder(model_name).encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    )


def probabilities(X: np.ndarray, coef: list[float], intercept: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(X @ np.asarray(coef) + intercept)))


def name_scores(names: list[str]) -> list[float] | None:
    """Вероятность «название похоже на раннюю технологию» для каждого имени или None."""
    art = artifact()
    if art is None or not names:
        return None
    try:
        with model_timer(step="name_model", model=art["encoder"], provider=PROVIDER):
            X = embed(names, art["encoder"])
    except Exception as exc:  # нет пакета, нет весов bge-m3, нехватка памяти — работаем без модели названий
        log.warning("Модель названий не сработала (%s) — оценка только по CatBoost", exc)
        return None
    return [float(p) for p in probabilities(X, art["coef"], art["intercept"])]


def blend_weight() -> float:
    art = artifact()
    return float(art["blend"]) if art else 0.0


def explain_ru(p: float) -> str:
    if p >= 0.6:
        return f"Название похоже на узкую раннюю технологию (модель названий: {p:.0%})"
    if p <= 0.3:
        return f"Название похоже на общее понятие, продукт или известный метод (модель названий: {p:.0%})"
    return f"По названию модель не уверена (модель названий: {p:.0%})"

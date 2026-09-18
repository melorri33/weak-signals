"""Офлайн-тесты разбора обучающей выборки (без файла организаторов)."""

import pytest

from src.model.dataset import load_negatives, normalize_stage


@pytest.mark.parametrize(
    ("raw", "stage"),
    [
        ("Концепция/Исследование", "research"),
        ("Исследование → Прототип/PoC", "research"),
        ("Прототип/PoC → Пилот", "prototype"),
        ("Пилот (controlled availability)", "prototype"),
        ("Раннее внедрение (у лидера) / Прототип (у остальных)", "product"),
        ("Ранние внедрения (ограниченный институциональный доступ)", "product"),
        ("что-то непонятное", None),
    ],
)
def test_normalize_stage(raw, stage):
    assert normalize_stage(raw) == stage


def test_negatives_are_well_formed():
    neg = load_negatives()
    assert len(neg) >= 80
    assert set(neg["kind"]) <= {"mature", "hype", "noise"}
    assert neg["term_en"].is_unique
    assert neg.notna().all().all()

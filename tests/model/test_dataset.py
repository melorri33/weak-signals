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
    assert len(neg) >= 200
    assert set(neg["kind"]) <= {"mature", "mainstream", "hype", "noise"}
    assert neg["term_en"].is_unique
    assert neg.notna().all().all()


def test_training_rebuilds_the_labeled_set_from_sources():
    """Обучение обязано пересобирать выборку, а не читать готовый файл.

    До 22.09 в train.main() стоял pd.read_csv готового data/labeled_set.csv, а
    build_labeled_set не вызывался ниоткуда. Из-за этого правка negatives.csv молча
    ни на что не влияла: добавили 74 примера и получили метрики до четвёртого знака
    те же самые. Тест ловит возврат к чтению готового файла.
    """
    import inspect

    from src.model import train

    source = inspect.getsource(train.main)
    assert "build_labeled_set" in source, "main() должна пересобирать выборку из negatives.csv и датасета"
    assert 'read_csv(DATA / "labeled_set.csv")' not in source, "main() снова читает готовую выборку"

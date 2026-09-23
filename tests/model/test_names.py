"""Офлайн-тесты модели названий: смешивание с CatBoost, объяснение, работа без модели."""

import importlib
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.common.schemas import Candidate, CandidateFeatures
from src.model import names, score
from src.model.train_names import CANDIDATE_LABELS, ERROR_KINDS, SYNTHETIC

score_module = importlib.import_module("src.model.score")  # модуль, а не одноимённая функция из src.model
REAL_NAME_SCORES = names.name_scores  # до подмены из tests/conftest.py
YEAR = date.today().year


@pytest.fixture
def fake_names(monkeypatch):
    probs = {"specific early tech": 0.9, "generic ai platform": 0.05}
    monkeypatch.setattr(names, "name_scores", lambda ns: [probs[n] for n in ns])
    monkeypatch.setattr(names, "blend_weight", lambda: 0.75)
    monkeypatch.setattr(score_module, "_load_model", lambda: None)  # эвристика вместо CatBoost


def _pair() -> tuple[list[Candidate], list[CandidateFeatures]]:
    """Два кандидата с одинаковыми публикациями: различаются только названием."""
    cands = [Candidate(id="g", name="generic ai platform"), Candidate(id="s", name="specific early tech")]
    same = {"total_pubs": 150, "growth_3y": 0.8, "first_seen_year": YEAR - 4}
    return cands, [CandidateFeatures(candidate_id=c.id, **same) for c in cands]


def test_name_model_lifts_specific_name_over_generic(fake_names):
    result = score(*_pair())

    assert [s.candidate_id for s in result] == ["s", "g"]
    specific, generic = result
    assert specific.top_reasons[0].feature == "name_pattern"
    assert "раннюю технологию" in specific.top_reasons[0].text
    assert generic.top_reasons[0].contribution < 0 < specific.top_reasons[0].contribution


def test_without_name_model_scores_stay_as_catboost(monkeypatch):
    monkeypatch.setattr(score_module, "_load_model", lambda: None)

    result = score(*_pair())

    assert result[0].score == result[1].score
    assert all(r.feature != "name_pattern" for s in result for r in s.top_reasons)


def test_missing_artifact_means_no_name_model(monkeypatch, tmp_path):
    monkeypatch.setattr(names, "NAME_MODEL_PATH", tmp_path / "missing.json")
    names.artifact.cache_clear()
    try:
        assert REAL_NAME_SCORES(["anything"]) is None
        assert names.blend_weight() == 0.0
    finally:
        names.artifact.cache_clear()


def test_explanation_by_probability():
    assert "раннюю технологию" in names.explain_ru(0.8)
    assert "общее понятие" in names.explain_ru(0.1)
    assert "не уверена" in names.explain_ru(0.45)


def test_probabilities_are_logistic():
    p = names.probabilities(np.array([[1.0, 0.0], [0.0, 1.0]]), coef=[2.0, -2.0], intercept=0.0)
    assert p[0] > 0.5 > p[1]


def test_training_files_are_consistent():
    cand = pd.read_csv(CANDIDATE_LABELS)
    syn = pd.read_csv(SYNTHETIC)
    assert set(cand["kind"]) <= ERROR_KINDS | {"weak_signal", "borderline", "off_topic"}
    assert cand["name"].notna().all() and syn["name"].notna().all()
    assert set(syn["label"]) == {0, 1}


@pytest.mark.skipif(not names.NAME_MODEL_PATH.exists(), reason="модель названий не обучена")
def test_artifact_fits_encoder():
    art = json.loads(names.NAME_MODEL_PATH.read_text(encoding="utf-8"))
    assert len(art["coef"]) == 1024  # размер эмбеддинга bge-m3
    assert 0 < art["blend"] < 1

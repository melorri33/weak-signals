"""Тексты интерфейса: показываем ровно то, что требует ТЗ, и ничего не придумываем."""

from __future__ import annotations

from datetime import date

import pytest

from src.common.schemas import (
    CONFIDENT_THRESHOLD,
    Explanation,
    SignalCard,
    SourceRef,
    SourceType,
    TrustLevel,
)
from src.ui import render


def _source(**kwargs) -> SourceRef:
    defaults = dict(
        document_id="techcrunch.com:abc123",
        title="Startup raises $12M for on-device inference",
        url="https://techcrunch.com/a",
        published=date(2026, 7, 8),
        source_type=SourceType.NEWS,
        language="en",
        trust=TrustLevel.MEDIUM,
    )
    return SourceRef(**{**defaults, **kwargs})


def _card(**kwargs) -> SignalCard:
    defaults = dict(
        candidate_id="on-device-inference",
        name="on-device inference",
        name_ru="Инференс на устройстве",
        score=0.81,
        description="описание",
        advantage="преимущество",
        case_example="кейс",
        why_weak_signal="почему ранний",
        sources=[_source()],
    )
    return SignalCard(**{**defaults, **kwargs})


def test_card_title_shows_both_names():
    assert render.card_title(_card()) == "Инференс на устройстве (on-device inference) — 0.81"


def test_card_title_without_russian_name():
    assert render.card_title(_card(name_ru=None)) == "on-device inference — 0.81"


def test_card_title_does_not_duplicate_same_name():
    card = _card(name="quantum sensing", name_ru="quantum sensing")
    assert render.card_title(card) == "quantum sensing — 0.81"


@pytest.mark.parametrize(
    ("score", "expected_start"),
    [(0.9, "Модель уверена"), (0.6, "Скорее ранняя"), (0.1, "Слабая оценка")],
)
def test_confidence_text_depends_on_score(score: float, expected_start: str):
    assert render.confidence_text(score, CONFIDENT_THRESHOLD).startswith(expected_start)


def test_reason_line_shows_direction_and_contribution():
    up = Explanation(feature="growth_3y", contribution=1.03, text="Публикации растут на 60% в год")
    down = Explanation(feature="total_pubs", contribution=-0.6, text="Всего публикаций: 3")

    assert render.reason_line(up) == "↑ Публикации растут на 60% в год (вклад +1.03)"
    assert render.reason_line(down) == "↓ Всего публикаций: 3 (вклад -0.60)"


def test_source_line_has_everything_the_spec_requires():
    """ТЗ: по каждому источнику — тип, дата, язык оригинала и уровень доверия."""
    line = render.source_line(_source())

    assert "новость" in line
    assert "08.07.2026" in line
    assert "английский" in line
    assert "доверие" in line


def test_source_without_date_says_so():
    assert "дата неизвестна" in render.source_line(_source(published=None))


def test_machine_made_texts_are_marked():
    """ТЗ: автоперевод и сгенерированное резюме помечаются рядом с источником."""
    marks = render.source_marks(_source(is_machine_translated=True, summary_is_generated=True))

    assert len(marks) == 2
    assert all("🤖" in mark for mark in marks)


def test_clean_source_has_no_marks():
    assert render.source_marks(_source()) == []


@pytest.mark.parametrize("code", ["mature", "hype", "noise"])
def test_excluded_labels_are_in_russian(code: str):
    label = render.excluded_label(code)
    assert label != code
    assert label[0].isupper()


def test_unknown_excluded_code_falls_back_to_itself():
    assert render.excluded_label("невиданное") == "невиданное"


def test_every_source_type_and_trust_level_has_a_label():
    """Новый тип источника или уровень доверия не должен показываться пользователю английским кодом."""
    assert set(render.SOURCE_TYPE_LABELS) == set(SourceType)
    assert set(render.TRUST_LABELS) == set(TrustLevel)

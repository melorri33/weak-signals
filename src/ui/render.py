"""Тексты интерфейса: как показать карточку, источник и причину отсева.

Вынесено из app.py отдельно, потому что это чистые функции — их видно в тестах без запуска Streamlit.
Здесь только оформление: ни один текст не придумывается, всё берётся из SearchResult.
"""

from __future__ import annotations

from datetime import date

from src.common.schemas import Explanation, SignalCard, SourceRef, SourceType, TrustLevel

TRUST_LABELS: dict[TrustLevel, str] = {
    TrustLevel.HIGH: "🟢 высокое доверие",
    TrustLevel.MEDIUM: "🟡 среднее доверие",
    TrustLevel.LOW: "🔴 низкое доверие",
}

SOURCE_TYPE_LABELS: dict[SourceType, str] = {
    SourceType.PAPER: "научная статья",
    SourceType.PREPRINT: "препринт",
    SourceType.PATENT: "патент",
    SourceType.NEWS: "новость",
    SourceType.REPORT: "аналитический отчёт",
    SourceType.GOV: "госорган или регулятор",
    SourceType.BLOG: "блог",
    SourceType.SOCIAL: "соцсеть",
    SourceType.PRESS_RELEASE: "пресс-релиз",
    SourceType.OTHER: "другое",
}

EXCLUDED_LABELS = {
    "mature": "Зрелая технология",
    "hype": "Медийная тема",
    "noise": "Недостаточно подтверждений",
    "no_research": "Нет следа в исследованиях",
    "ok": "Оставлено",
}

LANGUAGE_LABELS = {"ru": "русский", "en": "английский"}


def card_title(card: SignalCard) -> str:
    """«Название по-русски (Original name) — 0.82»; русское название есть не всегда."""
    name = f"{card.name_ru} ({card.name})" if card.name_ru and card.name_ru != card.name else card.name
    return f"{name} — {card.score:.2f}"


def confidence_text(score: float, confident_threshold: float) -> str:
    if score >= confident_threshold:
        return "Модель уверена, что это ранняя технология"
    if score >= 0.5:
        return "Скорее ранняя технология, но уверенности мало"
    return "Слабая оценка: показываем, потому что просили топ-15"


def reason_line(reason: Explanation) -> str:
    """«Публикации растут на 60% в год» → «↑ … (вклад +1.03)». Знак вклада показываем явно."""
    arrow = "↑" if reason.contribution > 0 else "↓"
    return f"{arrow} {reason.text} (вклад {reason.contribution:+.2f})"


def source_line(source: SourceRef) -> str:
    """Строка источника со всем, что требует ТЗ: тип, дата, язык, уровень доверия."""
    parts = [
        SOURCE_TYPE_LABELS.get(source.source_type, str(source.source_type)),
        _published_text(source.published),
        LANGUAGE_LABELS.get(source.language, source.language),
        TRUST_LABELS[source.trust],
    ]
    return " · ".join(p for p in parts if p)


def source_marks(source: SourceRef) -> list[str]:
    """Пометки к источнику: ТЗ требует показывать, что текст сделан машиной."""
    marks = []
    if source.is_machine_translated:
        marks.append("🤖 машинный перевод")
    if source.summary_is_generated:
        marks.append("🤖 резюме сгенерировано моделью")
    return marks


def _published_text(published: date | None) -> str:
    return published.strftime("%d.%m.%Y") if published else "дата неизвестна"


def excluded_label(reason_code: str) -> str:
    return EXCLUDED_LABELS.get(reason_code, reason_code)

"""Интерфейс сервиса: запрос → топ-15 ранних технологий с объяснением и источниками.

Запуск: streamlit run src/ui/app.py

Два режима:
  - «Новый поиск» — запускает конвейер целиком. На ноутбуке без видеокарты это долго (LLM считает
    на процессоре), поэтому прогресс показывается по шагам, а результат сохраняется в data/search_runs.
  - «Готовый прогон» — открывает сохранённый результат. Нужен для демонстрации жюри: показать
    выдачу сразу, не заставляя ждать прогон.

Интерфейс ничего не придумывает: все тексты, ссылки и оценки берутся из SearchResult.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from src.common.schemas import CONFIDENT_THRESHOLD, TOP_N, SearchResult, SignalCard
from src.ui.render import (
    card_title,
    confidence_text,
    excluded_label,
    reason_line,
    source_line,
    source_marks,
)

RUNS_DIR = Path("data") / "search_runs"
EXAMPLE_QUERIES = [
    "перспективные технологии edge AI и граничных вычислений",
    "перспективные решения в финтехе",
    "перспективные технологии в энергетике",
]


def main() -> None:
    st.set_page_config(page_title="Слабые сигналы", page_icon="📡", layout="wide")
    st.title("📡 Поиск слабых технологических сигналов")
    st.caption(
        f"Запрос свободной формы → топ-{TOP_N} зарождающихся технологий. "
        "У каждой — оценка модели, объяснение и источники. Зрелое, хайп и шум отсеиваются с причиной."
    )

    result = _sidebar_and_input()
    if result is None:
        st.info("Введите запрос и нажмите «Искать» — или откройте готовый прогон в левой панели.")
        return
    if result.status == "error":
        st.error(f"Прогон не удался: {result.error}")
        return
    _show_result(result)


def _sidebar_and_input() -> SearchResult | None:
    with st.sidebar:
        st.header("Готовые прогоны")
        saved = sorted(RUNS_DIR.glob("*.json")) if RUNS_DIR.exists() else []
        if saved:
            choice = st.selectbox("Открыть сохранённый результат", ["—", *[p.stem for p in saved]])
            if choice != "—":
                path = next(p for p in saved if p.stem == choice)
                return SearchResult.model_validate_json(path.read_text(encoding="utf-8"))
        else:
            st.caption("Сохранённых прогонов пока нет.")

    query = st.text_input("Что ищем?", placeholder=EXAMPLE_QUERIES[0])
    st.caption("Например: " + " · ".join(f"«{q}»" for q in EXAMPLE_QUERIES))
    if not st.button("Искать", type="primary") or not query.strip():
        return st.session_state.get("last_result")

    result = _run_pipeline(query.strip())
    st.session_state["last_result"] = result
    return result


def _run_pipeline(query: str) -> SearchResult:
    """Прогон с показом шага: на процессоре это единицы минут, без прогресса выглядит как зависание."""
    from src.pipeline.run import run  # импорт здесь: открытие сохранённого прогона не тянет конвейер

    status = st.status("Запускаем конвейер…", expanded=True)

    def on_progress(partial: SearchResult) -> None:
        status.update(label=f"Шаг: {partial.stage}")
        status.write(
            f"{partial.stage}: документов {partial.documents_processed}, кандидатов {partial.candidates_found}"
        )

    result = asyncio.run(run(query, on_progress=on_progress))
    status.update(label=f"Готово за {result.duration_s or 0:.0f} с", state="complete", expanded=False)
    _save(result)
    return result


def _save(result: SearchResult) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{result.run_id}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _show_result(result: SearchResult) -> None:
    st.subheader(f"«{result.query}»")
    cols = st.columns(4)
    cols[0].metric("Документов", result.documents_processed)
    cols[1].metric("Кандидатов", result.candidates_found)
    cols[2].metric("В выдаче", len(result.top))
    cols[3].metric(f"Уверенных (>{CONFIDENT_THRESHOLD})", result.confident_signals)

    signals, excluded, others, models = st.tabs(
        [
            f"Сигналы ({len(result.top)})",
            f"Отсеяно ({len(result.excluded)})",
            f"Все кандидаты ({len(result.scored)})",
            f"Журнал моделей ({len(result.model_calls)})",
        ]
    )
    with signals:
        _show_cards(result.top)
    with excluded:
        _show_excluded(result)
    with others:
        _show_scored(result)
    with models:
        _show_model_calls(result)

    if result.expanded_phrases:
        with st.expander(f"Поисковые фразы ({len(result.expanded_phrases)})"):
            st.write(", ".join(result.expanded_phrases))


def _show_cards(cards: list[SignalCard]) -> None:
    if not cards:
        st.warning("Ни одной технологии в выдаче.")
        return
    for i, card in enumerate(cards, start=1):
        with st.expander(f"{i}. {card_title(card)}", expanded=i <= 3):
            st.progress(card.score, text=confidence_text(card.score, CONFIDENT_THRESHOLD))
            st.write(card.description)
            st.markdown(f"**Преимущество.** {card.advantage}")
            st.markdown(f"**Где пробуют.** {card.case_example}")
            st.markdown(f"**Почему ранний сигнал.** {card.why_weak_signal}")
            if card.top_reasons:
                st.markdown("**Что повлияло на оценку**")
                for reason in card.top_reasons:
                    st.markdown(f"- {reason_line(reason)}")
            _show_sources(card)
            if card.report_md:
                with st.expander("Подробный отчёт"):
                    st.markdown(card.report_md)


def _show_sources(card: SignalCard) -> None:
    st.markdown(f"**Источники ({len(card.sources)})**")
    for source in card.sources:
        st.markdown(f"- [{source.title}]({source.url})  \n  {source_line(source)}")
        if marks := source_marks(source):
            st.caption("  " + " · ".join(marks))
        if source.ru_summary:
            st.caption(f"  {source.ru_summary}")


def _show_excluded(result: SearchResult) -> None:
    if not result.excluded:
        st.caption("Правила отсева никого не исключили.")
        return
    st.caption("Технологии, которые не попали в выдачу, и причина — требование ТЗ.")
    st.dataframe(
        [
            {"Технология": d.name, "Причина": excluded_label(d.reason_code), "Пояснение": d.reason_text}
            for d in result.excluded
        ],
        hide_index=True,
        width="stretch",
    )


def _show_scored(result: SearchResult) -> None:
    if not result.scored:
        st.caption("Список кандидатов пуст.")
        return
    st.caption("Все кандидаты после отсева с оценкой модели: видно, что нашли, но ранжировали низко.")
    st.dataframe(
        [{"Технология": s.name, "Оценка": round(s.score, 3)} for s in result.scored],
        hide_index=True,
        width="stretch",
    )


def _show_model_calls(result: SearchResult) -> None:
    if not result.model_calls:
        st.caption("Вызовов моделей не было.")
        return
    st.caption("Какая модель на каком шаге вызывалась и сколько заняла — требование ТЗ.")
    st.dataframe(
        [
            {"Шаг": c.step, "Модель": c.model, "Провайдер": c.provider, "Время, с": round(c.duration_ms / 1000, 1)}
            for c in result.model_calls
        ],
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Скачать результат в JSON",
        data=json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        file_name=f"{result.run_id}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()

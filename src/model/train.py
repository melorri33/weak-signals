"""Обучение модели «слабый сигнал / нет»: CatBoost + SHAP, отчёт P/R/F1.

Признаки идут тем же путём, что и в конвейере: статистика термина → TermStats → features.compute → vectorize.
Статистика для обучающей выборки берётся из кэшей экспериментов (notebooks/01, 02):
  data/openalex_years_phrase.json — публикации OpenAlex по годам (поиск точной фразы)
  data/attention_stats.json   — Hacker News по годам (медиа), наличие статьи в Википедии

Запуск: python -m src.model.train
Результат: src/model/artifacts/weak_signal.cbm, reports/metrics.md, reports/metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.common.schemas import Candidate, TermStats
from src.features import compute
from src.model.vectorize import FEATURE_NAMES, FEATURES, to_row

DATA = Path("data")
ARTIFACTS = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS / "weak_signal.cbm"
REPORTS = Path("reports")
SEED = 42
THRESHOLD = 0.5

CATBOOST_PARAMS = {
    "iterations": 400,
    "depth": 4,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "auto_class_weights": "Balanced",
    "random_seed": SEED,
    "verbose": False,
    "allow_writing_files": False,
}


def load_term_stats() -> dict[str, TermStats]:
    """Собрать TermStats по терминам обучающей выборки из кэшей экспериментов."""
    pubs = json.loads((DATA / "openalex_years_phrase.json").read_text(encoding="utf-8"))
    attention = json.loads((DATA / "attention_stats.json").read_text(encoding="utf-8"))
    stats = {}
    for term, years in pubs.items():
        att = attention.get(term)
        stats[term] = TermStats(
            term=term,
            pubs_by_year={int(y): n for y, n in years.items()},
            news_by_year={int(y): n for y, n in att["hn"].items()} if att else None,
            wikipedia_en=(att["wiki"]["article"] is not None) if att else None,
        )
    return stats


def build_matrix(labeled: pd.DataFrame, stats: dict[str, TermStats]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Матрица признаков через features.compute — как в конвейере."""
    rows, keep = [], []
    for i, r in labeled.iterrows():
        st = stats.get(r["term_en"])
        if st is None:
            continue
        feats = compute(Candidate(id=r["key"], name=r["term_en"]), [], st)
        rows.append(to_row(feats))
        keep.append(i)
    X = pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=float)
    kept = labeled.loc[keep].reset_index(drop=True)
    return X, kept["label"].to_numpy(), kept


def cross_validate(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Out-of-fold вероятности, 5-fold stratified CV."""
    oof = np.zeros(len(y))
    for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return oof


def metrics(y: np.ndarray, proba: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def shap_importance(model: CatBoostClassifier, X: pd.DataFrame) -> pd.Series:
    """Средний |SHAP| по признакам (глобальная важность)."""
    shap = model.get_feature_importance(Pool(X), type="ShapValues")[:, :-1]
    return pd.Series(np.abs(shap).mean(axis=0), index=[f.title_ru for f in FEATURES]).sort_values(ascending=False)


def write_report(m: dict[str, float], by_kind: pd.Series, importance: pd.Series, n: int, pos: int) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(
        json.dumps({k: round(v, 4) for k, v in m.items()}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Метрики модели «слабый сигнал / нет»",
        "",
        f"Выборка: {n} технологий, из них {pos} слабых сигналов (датасет организаторов) и {n - pos} отрицательных",
        "(`src/model/training/negatives.csv`). Оценка — 5-fold stratified кросс-валидация, порог 0.5.",
        "",
        "| Метрика | Значение |",
        "| --- | --- |",
        *[f"| {k} | {v:.3f} |" for k, v in m.items()],
        "",
        "## Точность по типам примеров",
        "",
        "| Тип | Доля верных |",
        "| --- | --- |",
        *[f"| {k} | {v:.2f} |" for k, v in by_kind.items()],
        "",
        "## Важность признаков (средний |SHAP|)",
        "",
        "| Признак | Важность |",
        "| --- | --- |",
        *[f"| {k} | {v:.3f} |" for k, v in importance.items()],
        "",
    ]
    (REPORTS / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    labeled = pd.read_csv(DATA / "labeled_set.csv")
    X, y, kept = build_matrix(labeled, load_term_stats())
    oof = cross_validate(X, y)
    m = metrics(y, oof)
    correct = pd.Series((oof >= THRESHOLD).astype(int) == y, index=kept.index)
    by_kind = correct.groupby(kept["kind"]).mean().round(2)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X, y)
    ARTIFACTS.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    importance = shap_importance(model, X)
    write_report(m, by_kind, importance, len(y), int(y.sum()))

    print(pd.Series(m).round(3).to_string())
    print("\nПо типам:\n" + by_kind.to_string())
    print("\nВажность (|SHAP|):\n" + importance.round(3).to_string())
    kept["proba"] = oof
    print("\nОшибки с наибольшей уверенностью:")
    print(kept[(oof >= THRESHOLD) != y.astype(bool)].nlargest(8, "proba")[["term_en", "kind", "proba"]].to_string())
    print(f"\nМодель: {MODEL_PATH}")


if __name__ == "__main__":
    main()

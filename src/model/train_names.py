"""Обучение модели названий (src/model/names.py): `python -m src.model.train_names [--check]`.

Положительные примеры — термины датасета организаторов (data/, не коммитим), слабые сигналы из
ручной разметки кандидатов и синтетические ранние технологии из областей вне датасета.
Отрицательные — training/negatives.csv, ошибки из разметки (слишком широко, зрелое, не технология,
продукт) и синтетические ошибки. «Спорно» и «не по теме» в обучение не идут: не по теме —
свойство пары «название + запрос», а не самого названия.

--check: честная проверка по областям. Для каждой области модель учится без её разметки и без
её терминов датасета и оценивает кандидатов этой области (ROC AUC: слабый сигнал против ошибки).
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from src.common.config import get_settings
from src.model.dataset import TRAINING_DIR, load_negatives, load_signals
from src.model.names import NAME_MODEL_PATH, embed
from src.model.train import POSITIVE_TERMS, SIGNALS_XLSX

CANDIDATE_LABELS = TRAINING_DIR / "candidate_labels.csv"
SYNTHETIC = TRAINING_DIR / "synthetic_names.csv"
# Доля модели названий в итоговой уверенности. Подбор 23.09 на трёх прогонах (270 мест топ-15,
# ручная разметка): 0.6 → 45 слабых сигналов, 0.75 → 47, 0.8 → 48, 0.9 → 43, только CatBoost → 40.
BLEND = 0.75
C = 1.0
ERROR_KINDS = {"too_broad", "mature", "not_a_technology", "product"}


def training_table() -> pd.DataFrame:
    """name, label (1/0), domain (пусто у синтетики), source."""
    pos = load_signals(SIGNALS_XLSX, POSITIVE_TERMS)
    neg = load_negatives()
    cand = pd.read_csv(CANDIDATE_LABELS)
    cand = cand[cand["kind"].isin(ERROR_KINDS | {"weak_signal"})]
    syn = pd.read_csv(SYNTHETIC)
    parts = [
        pd.DataFrame({"name": pos["term_en"], "label": 1, "domain": pos["domain"], "source": "dataset"}),
        pd.DataFrame({"name": neg["term_en"], "label": 0, "domain": neg["domain"], "source": "negatives"}),
        pd.DataFrame(
            {
                "name": cand["name"],
                "label": (cand["kind"] == "weak_signal").astype(int),
                "domain": cand["domain"],
                "source": "candidates",
            }
        ),
        pd.DataFrame({"name": syn["name"], "label": syn["label"], "domain": None, "source": "synthetic"}),
    ]
    return pd.concat(parts, ignore_index=True).dropna(subset=["name"]).reset_index(drop=True)


def fit(X: np.ndarray, y: np.ndarray):  # -> LogisticRegression
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)


def check(table: pd.DataFrame, X: np.ndarray) -> None:
    from sklearn.metrics import roc_auc_score

    y_all, p_all = [], []
    for domain in sorted(table.loc[table["source"] == "candidates", "domain"].unique()):
        train = (table["domain"] != domain).to_numpy()
        test = ((table["domain"] == domain) & (table["source"] == "candidates")).to_numpy()
        clf = fit(X[train], table["label"].to_numpy()[train])
        p = clf.predict_proba(X[test])[:, 1]
        y = table["label"].to_numpy()[test]
        y_all += list(y)
        p_all += list(p)
        print(f"  {domain:20} ROC AUC {roc_auc_score(y, p):.3f}  (слабых сигналов {int(y.sum())} из {len(y)})")
    print(f"  {'все области':20} ROC AUC {roc_auc_score(y_all, p_all):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="проверка по областям, без сохранения модели")
    args = ap.parse_args()

    encoder = get_settings().embed_model
    table = training_table()
    print(f"Примеров: {len(table)}, положительных {int(table['label'].sum())}; по источникам:")
    print(table.groupby(["source", "label"]).size().to_string())
    X = embed(table["name"].tolist(), encoder)
    if args.check:
        check(table, X)
        return
    clf = fit(X, table["label"].to_numpy())
    NAME_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    NAME_MODEL_PATH.write_text(
        json.dumps(
            {
                "encoder": encoder,
                "blend": BLEND,
                "intercept": round(float(clf.intercept_[0]), 6),
                "coef": [round(float(v), 6) for v in clf.coef_[0]],
                "trained": date.today().isoformat(),
                "examples": int(len(table)),
                "positives": int(table["label"].sum()),
            }
        ),
        encoding="utf-8",
    )
    print(f"Модель названий сохранена: {NAME_MODEL_PATH}")


if __name__ == "__main__":
    main()

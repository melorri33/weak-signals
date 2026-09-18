"""Эксперимент: отличаются ли слабые сигналы от зрелых технологий по динамике публикаций в OpenAlex.

Не продакшен-код (настоящий term_stats — в src/collectors, у Данных). Здесь быстрая проверка гипотезы.
Вход: data/labeled_set.csv (собирается src.model.dataset.build_labeled_set).
Кэш ответов: data/openalex_years.json — повторный запуск не ходит в сеть.

Запуск: python notebooks/01_openalex_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
CACHE = DATA / "openalex_years.json"
API = "https://api.openalex.org/works"
LAST_FULL_YEAR = 2025  # 2026 ещё не закончился — рост считаем по полным годам


def fetch_years(term: str, client: httpx.Client) -> dict[int, int]:
    """Число работ по годам, где термин встречается в заголовке или аннотации."""
    params = {"filter": f"title_and_abstract.search:{term}", "group_by": "publication_year", "per_page": 200}
    r = client.get(API, params=params, timeout=20)
    r.raise_for_status()
    return {int(g["key"]): g["count"] for g in r.json()["group_by"] if g["key"].isdigit()}


def load_or_fetch(terms: list[str]) -> dict[str, dict[int, int]]:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [t for t in terms if t not in cache]
    with httpx.Client(headers={"User-Agent": "weak-signals-hackathon (research probe)"}) as client:
        for i, term in enumerate(todo, 1):
            try:
                cache[term] = fetch_years(term, client)
            except httpx.HTTPError as e:
                print(f"  ошибка на «{term}»: {e}")
                continue
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
                CACHE.write_text(json.dumps(cache), encoding="utf-8")
            time.sleep(0.15)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return {t: {int(y): n for y, n in cache[t].items()} for t in terms if t in cache}


def features(years: dict[int, int]) -> dict[str, float]:
    total = sum(years.values())
    last = years.get(LAST_FULL_YEAR, 0)
    base = years.get(LAST_FULL_YEAR - 3, 0)
    recent = sum(n for y, n in years.items() if y >= LAST_FULL_YEAR - 2)
    first = min((y for y, n in years.items() if n >= 3), default=LAST_FULL_YEAR + 1)
    return {
        "log_total": np.log1p(total),
        "log_last_year": np.log1p(last),
        "growth_3y": np.log1p(last) - np.log1p(base),  # лог-рост: устойчив к нулям
        "recent_share": recent / total if total else 0.0,
        "age_years": LAST_FULL_YEAR + 1 - first,
    }


def main() -> None:
    df = pd.read_csv(DATA / "labeled_set.csv")
    years = load_or_fetch(df["term_en"].tolist())
    df = df[df["term_en"].isin(years)].reset_index(drop=True)
    X = pd.DataFrame([features(years[t]) for t in df["term_en"]])
    y = df["label"].to_numpy()

    print(f"\nПримеров: {len(df)} (сигналов {y.sum()}, не-сигналов {len(y) - y.sum()})")
    print("\nМедианы признаков по классам:")
    print(pd.concat([X, df[["kind"]]], axis=1).groupby("kind").median().round(2).to_string())

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\nЛогистическая регрессия, 5-fold CV (только признаки публикаций):")
    auc, pr_auc = roc_auc_score(y, proba), average_precision_score(y, proba)
    print(f"  ROC-AUC {auc:.3f} | PR-AUC {pr_auc:.3f} (база {y.mean():.2f})")
    print(f"  F1 {f1_score(y, pred):.3f} | accuracy {(pred == y).mean():.3f}")
    model.fit(X, y)
    coefs = pd.Series(model[-1].coef_[0], index=X.columns).sort_values()
    print("\nВеса признаков (стандартизованные; >0 — за сигнал):")
    print(coefs.round(2).to_string())

    df["proba"] = proba
    print("\nСамые уверенные ошибки:")
    fn = df[(y == 1)].nsmallest(5, "proba")[["term_en", "proba"]]
    fp = df[(y == 0)].nlargest(5, "proba")[["term_en", "kind", "proba"]]
    print("  сигналы, которые модель не узнала:\n", fn.to_string(index=False))
    print("  не-сигналы, принятые за сигнал:\n", fp.to_string(index=False))


if __name__ == "__main__":
    main()

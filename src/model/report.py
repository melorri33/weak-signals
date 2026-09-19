"""Отчёт об оценке модели для сдачи: графики в reports/figures/ и reports/model_report.md.

Все числа — out-of-fold (5-fold stratified CV), как в reports/metrics.md; SHAP — по итоговой модели.
Запуск: python -m src.model.report  (нужны кэши data/, как для src.model.train)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import Pool  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.model import train as T  # noqa: E402
from src.model.vectorize import FEATURES  # noqa: E402

FIG = T.REPORTS / "figures"
KIND_RU = {
    "weak_signal": "слабые сигналы",
    "mature": "зрелые",
    "mainstream": "массовые растущие",
    "hype": "хайп",
    "noise": "шум",
}
PUBLICATION_FEATURES = ["log_total_pubs", "log_pubs_last_year", "growth_3y", "recent_share", "age_years"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig: plt.Figure, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)
    return f"figures/{name}"


def confusion(y: np.ndarray, proba: np.ndarray) -> str:
    cm = confusion_matrix(y, proba >= T.THRESHOLD)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.imshow(cm, cmap="Blues")
    labels = ["не сигнал", "слабый сигнал"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("предсказание модели")
    ax.set_ylabel("разметка")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_title("Матрица ошибок (5-fold CV)")
    return _save(fig, "confusion_matrix.png")


def curves(y: np.ndarray, proba: np.ndarray, baseline: np.ndarray) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.4))
    for p, label in [(proba, "CatBoost, все признаки"), (baseline, "логрегрессия, только публикации")]:
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, label=label)
        prec, rec, _ = precision_recall_curve(y, p)
        b.plot(rec, prec, label=label)
    a.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    a.set(xlabel="доля ложных срабатываний", ylabel="доля найденных сигналов", title="ROC-кривая")
    b.axhline(y.mean(), ls="--", c="grey", lw=0.8)
    b.set(xlabel="полнота (recall)", ylabel="точность (precision)", title="Precision–Recall")
    a.legend(fontsize=8, loc="lower right")
    return _save(fig, "roc_pr.png")


def shap_summary(X: pd.DataFrame, y: np.ndarray) -> str:
    """Вклад каждого признака для каждого примера: цвет — значение признака, ось — вклад в «сигнал»."""
    model = T.fit(X, y)
    Xi = T.apply_impute(X, T.impute_values(X))
    shap = model.get_feature_importance(Pool(Xi), type="ShapValues")[:, :-1]
    order = np.argsort(np.abs(shap).mean(axis=0))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        vals = Xi.iloc[:, j].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        colors = np.full(len(vals), 0.5)
        if finite.any() and np.ptp(vals[finite]) > 0:
            colors[finite] = (vals[finite] - np.nanmin(vals)) / np.ptp(vals[finite])
        ax.scatter(shap[:, j], row + rng.uniform(-0.25, 0.25, len(vals)), c=colors, cmap="coolwarm", s=8, alpha=0.8)
    ax.set_yticks(range(len(order)), [FEATURES[j].title_ru for j in order])
    ax.axvline(0, c="grey", lw=0.8)
    ax.set_xlabel(
        "вклад SHAP: > 0 — за слабый сигнал, < 0 — против\nцвет точки: синий — значение признака мало, красный — много"
    )
    ax.set_title("Как признаки влияют на решение модели")
    return _save(fig, "shap_summary.png")


def distributions(X: pd.DataFrame, kinds: pd.Series) -> str:
    show = [
        ("log_pubs_last_year", "Публикаций за последний год (лог)"),
        ("preprint_share", "Доля препринтов"),
        ("recent_share", "Доля работ за 3 года"),
        ("log_news_total", "Упоминаний в техмедиа (лог)"),
    ]
    fig, axes = plt.subplots(1, len(show), figsize=(12, 3.2), sharey=True)
    order = ["weak_signal", "hype", "mainstream", "mature", "noise"]
    for ax, (col, title) in zip(axes, show, strict=True):
        data = [X.loc[kinds == k, col].dropna().to_numpy() for k in order]
        ax.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False)
        ax.set_title(title, fontsize=9)
    axes[0].set_yticks(range(1, len(order) + 1), [KIND_RU[k] for k in order])
    return _save(fig, "feature_distributions.png")


def baseline_oof(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    Xb = X[PUBLICATION_FEATURES].fillna(X[PUBLICATION_FEATURES].median())
    cv = StratifiedKFold(5, shuffle=True, random_state=T.SEED)
    return cross_val_predict(model, Xb, y, cv=cv, method="predict_proba")[:, 1]


def main() -> None:
    labeled = pd.read_csv(T.DATA / "labeled_set.csv")
    X, y, kept = T.build_matrix(labeled, T.load_term_stats())
    oof = T.cross_validate(X, y)
    base = baseline_oof(X, y)
    m, mb = T.metrics(y, oof), T.metrics(y, base)
    by_kind = pd.Series((oof >= T.THRESHOLD).astype(int) == y).groupby(kept["kind"].to_numpy()).mean()

    figs = {
        "cm": confusion(y, oof),
        "curves": curves(y, oof, base),
        "shap": shap_summary(X, y),
        "dist": distributions(X, kept["kind"]),
    }
    rows = [f"| {k} | {m[k]:.3f} | {mb[k]:.3f} |" for k in m]
    kinds = [f"| {KIND_RU[k]} | {int((kept['kind'] == k).sum())} | {v:.0%} |" for k, v in by_kind.items()]
    md = [
        "# Отчёт об оценке модели «слабый сигнал / нет»",
        "",
        f"Выборка: {len(y)} технологий — {int(y.sum())} слабых сигналов из датасета организаторов и "
        f"{len(y) - int(y.sum())} отрицательных примеров команды (`src/model/training/negatives.csv`).",
        "Оценка — 5-fold stratified кросс-валидация: каждый пример проверяется моделью, которая его не видела.",
        "Главная метрика по ответу организаторов — accuracy. Методология — `docs/methodology.md`.",
        "",
        "## Метрики",
        "",
        "| Метрика | CatBoost (итоговая модель) | Базовая линия: логрегрессия на публикациях |",
        "| --- | --- | --- |",
        *rows,
        "",
        f"![Матрица ошибок]({figs['cm']})",
        "",
        f"![ROC и PR]({figs['curves']})",
        "",
        "## Точность по типам примеров",
        "",
        "| Тип | Примеров | Верных ответов |",
        "| --- | --- | --- |",
        *kinds,
        "",
        "## Чем слабые сигналы отличаются от остального",
        "",
        f"![Распределения признаков]({figs['dist']})",
        "",
        "## Объяснимость: вклад признаков (SHAP)",
        "",
        "Каждая точка — одна технология. Справа от нуля — признак толкает к «слабому сигналу», слева — против.",
        "",
        f"![SHAP]({figs['shap']})",
        "",
    ]
    (T.REPORTS / "model_report.md").write_text("\n".join(md), encoding="utf-8")
    print(pd.DataFrame({"CatBoost": m, "база": mb}).round(3).to_string())
    print(f"\nОтчёт: {T.REPORTS / 'model_report.md'}")


if __name__ == "__main__":
    main()

"""Эксперимент: признаки внимания медиа и энциклопедий для обучающей выборки.

Не продакшен-код (настоящий term_stats — в src/collectors, у Данных). Проверяем, какие источники помогают
отличить хайп и зрелое от слабых сигналов.

- Hacker News (Algolia API, без ключа): число историй с термином по годам — внимание техмедиа.
- Википедия (en): есть ли статья о термине, сколько страниц его упоминают, просмотры статьи за 2025 год.

GDELT на 18.09.2026 отвечал 429 даже с паузой 5+ с — поэтому новости берём из HN.
Кэш: data/attention_stats.json. Запуск: python notebooks/02_attention_stats.py
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

DATA = Path("data")
CACHE = DATA / "attention_stats.json"
# Википедия требует контакт в User-Agent, иначе 403.
UA = {"User-Agent": "weak-signals/0.1 (https://github.com/melorri33/weak-signals)"}
HN_API = "https://hn.algolia.com/api/v1/search"
WIKI_API = "https://en.wikipedia.org/w/api.php"
PAGEVIEWS = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{}/monthly/20250101/20251231"
PAUSE_S = 0.3  # пауза между запросами к HN


def _year_ts(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=UTC).timestamp())


def hn_by_year(term: str, client: httpx.Client) -> dict[int, int]:
    """Число историй HN с точной фразой: 2016–2022 одним числом (ключ 2016) и по годам с 2023.

    Пять запросов вместо одиннадцати: при частых запросах HN отвечает 403. Суммы те же, что по годам.
    """
    buckets = [(2016, 2023), (2023, 2024), (2024, 2025), (2025, 2026), (2026, 2027)]
    out = {}
    for start, end in buckets:
        flt = f"created_at_i>={_year_ts(start)},created_at_i<{_year_ts(end)}"
        r = client.get(HN_API, params={"query": f'"{term}"', "tags": "story", "hitsPerPage": 0, "numericFilters": flt})
        r.raise_for_status()
        out[start] = r.json()["nbHits"]
        time.sleep(PAUSE_S)
    return out


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def wikipedia(term: str, client: httpx.Client) -> dict[str, object]:
    """Есть ли статья с названием, почти совпадающим с термином; сколько страниц его упоминают."""
    r = client.get(
        WIKI_API,
        params={"action": "query", "list": "search", "srsearch": f'"{term}"', "srlimit": 5, "format": "json"},
    )
    r.raise_for_status()
    q = r.json()["query"]
    term_tokens = _tokens(term)
    article = None
    for hit in q["search"]:
        title_tokens = _tokens(hit["title"])
        overlap = len(term_tokens & title_tokens) / max(len(term_tokens | title_tokens), 1)
        if overlap >= 0.6:
            article = hit["title"]
            break
    views = None
    if article:
        pv = client.get(PAGEVIEWS.format(article.replace(" ", "_")))
        if pv.status_code == 200:
            views = sum(i["views"] for i in pv.json()["items"])
    return {"article": article, "mentions": q["searchinfo"]["totalhits"], "views_2025": views}


def _collect_one(term: str, client: httpx.Client) -> dict[str, object]:
    return {"hn": hn_by_year(term, client), "wiki": wikipedia(term, client)}


def main() -> None:
    terms = pd.read_csv(DATA / "labeled_set.csv")["term_en"].tolist()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [t for t in terms if t not in cache]
    print(f"Нужно собрать: {len(todo)} из {len(terms)}")
    # Последовательно: HN блокирует параллельные запросы (403).
    with httpx.Client(headers=UA, timeout=20) as client:
        for i, term in enumerate(todo, 1):
            try:
                cache[term] = _collect_one(term, client)
            except httpx.HTTPError as e:
                print(f"  ошибка на «{term}»: {e}", flush=True)
                time.sleep(10)
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"Готово: {len(cache)} терминов в кэше")


if __name__ == "__main__":
    main()

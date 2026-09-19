"""Эксперимент: нечёткий поиск в Hacker News (все слова термина, без точной фразы).

Гипотеза: хайп-термины из модных слов («AI copilot for engineers») по точной фразе почти не встречаются,
а по словам — очень часто; у настоящих ранних технологий разница маленькая.
Кэш: data/hn_loose.json → {term: {"2016": до 2023, "2023": с 2023}}. Запуск: python notebooks/03_hn_loose.py
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

DATA = Path("data")
CACHE = DATA / "hn_loose.json"
UA = {"User-Agent": "weak-signals/0.1 (https://github.com/melorri33/weak-signals)"}
HN_API = "https://hn.algolia.com/api/v1/search"
PAUSE_S = 0.3
BUCKETS = [(2016, 2023), (2023, 2027)]


def _ts(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=UTC).timestamp())


def hn_loose(term: str, client: httpx.Client) -> dict[int, int]:
    out = {}
    for start, end in BUCKETS:
        flt = f"created_at_i>={_ts(start)},created_at_i<{_ts(end)}"
        r = client.get(HN_API, params={"query": term, "tags": "story", "hitsPerPage": 0, "numericFilters": flt})
        r.raise_for_status()
        out[start] = r.json()["nbHits"]
        time.sleep(PAUSE_S)
    return out


def main() -> None:
    terms = pd.read_csv(DATA / "labeled_set.csv")["term_en"].tolist()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [t for t in terms if t not in cache]
    print(f"Нужно собрать: {len(todo)} из {len(terms)}", flush=True)
    with httpx.Client(headers=UA, timeout=20) as client:
        for i, term in enumerate(todo, 1):
            try:
                cache[term] = hn_loose(term, client)
            except httpx.HTTPError as e:
                print(f"  ошибка на «{term}»: {e}", flush=True)
                time.sleep(10)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
                CACHE.write_text(json.dumps(cache), encoding="utf-8")
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"Готово: {len(cache)}", flush=True)


if __name__ == "__main__":
    main()

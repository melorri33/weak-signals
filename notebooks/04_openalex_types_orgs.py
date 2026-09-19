"""Эксперимент: доля препринтов и число организаций по термину (OpenAlex, поиск точной фразы).

- Доля препринтов: у ранних технологий много препринтов (arXiv и т.п.), у зрелых — статьи и обзоры.
- Число организаций (authorships.institutions.lineage): мало разных организаций — нет широкого круга игроков.
  OpenAlex отдаёт не больше 200 групп, поэтому 200 означает «200 и больше».

Кэш: data/openalex_types_orgs.json.
Запуск из корня репозитория: PYTHONPATH=. python notebooks/04_openalex_types_orgs.py (ключ — из .env)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd

from src.common.config import get_settings

DATA = Path("data")
CACHE = DATA / "openalex_types_orgs.json"
API = "https://api.openalex.org/works"


def group_counts(term: str, group_by: str, client: httpx.Client) -> tuple[dict[str, int], int]:
    params = {"filter": f'title_and_abstract.search:"{term}"', "group_by": group_by, "per_page": 200}
    if key := get_settings().openalex_api_key:
        params["api_key"] = key
    r = client.get(API, params=params)
    r.raise_for_status()
    j = r.json()
    return {g["key_display_name"]: g["count"] for g in j["group_by"]}, j["meta"]["groups_count"]


def main() -> None:
    terms = pd.read_csv(DATA / "labeled_set.csv")["term_en"].tolist()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [t for t in terms if t not in cache]
    print(f"Нужно собрать: {len(todo)} из {len(terms)}", flush=True)
    with httpx.Client(timeout=30) as client:
        for i, term in enumerate(todo, 1):
            try:
                types, _ = group_counts(term, "type", client)
                _, n_orgs = group_counts(term, "authorships.institutions.lineage", client)
            except httpx.HTTPError as e:
                print(f"  ошибка на «{term}»: {e}", flush=True)
                continue
            cache[term] = {"types": types, "n_orgs": n_orgs}
            time.sleep(0.1)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
                CACHE.write_text(json.dumps(cache), encoding="utf-8")
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"Готово: {len(cache)}", flush=True)


if __name__ == "__main__":
    main()

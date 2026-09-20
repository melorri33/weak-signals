"""Проверка требования ТЗ: названий технологий из датасета организаторов нет в репозитории.

ТЗ запрещает ограничивать поиск датасетом или заранее заданным списком. Репозиторий публичный,
поэтому термин из `data/positive_terms.csv`, попавший в код, тест, промпт или комментарий, —
это нарушение, даже если он оказался там случайно, как пример или кусок адреса.

Тест пропускается, если `data/positive_terms.csv` нет: файл не в git, он есть только у тех,
кому передали данные. Пропуск — не «всё хорошо», а «проверить нечем».
"""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import pytest

TERMS_PATH = Path("data/positive_terms.csv")

# Двоичные и сгенерированные файлы: искать термины в них бессмысленно.
SKIP_SUFFIXES = (".png", ".jpg", ".cbm", ".xlsx", ".ico", ".pdf")

# Термин может быть разорван переносом строки, разделён дефисами в адресе или расставлен
# по markdown-разметке. Поэтому перед поиском схлопываем всё это в обычные пробелы:
# «neuromorphic-chips» в ссылке и «neuromorphic\n# chips» в комментарии — такие же нарушения,
# как термин, написанный напрямую.
_SEPARATORS = re.compile(r"[\s#>*_\-/]+")


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return [Path(line) for line in out.split("\n") if line and not line.endswith(SKIP_SUFFIXES)]


def _dataset_terms() -> list[str]:
    with TERMS_PATH.open(encoding="utf-8") as f:
        return [row["term_en"].strip() for row in csv.DictReader(f) if row["term_en"].strip()]


@pytest.mark.skipif(not TERMS_PATH.exists(), reason="нет data/positive_terms.csv — проверять нечем")
def test_dataset_terms_are_not_in_repository():
    terms = _dataset_terms()
    assert terms, "файл терминов пуст — проверка не имеет смысла"
    patterns = [(t, re.compile(rf"(?<!\w){re.escape(t.lower())}(?!\w)")) for t in terms]

    violations: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        flat = _SEPARATORS.sub(" ", text)
        violations += [f"{path}: «{term}»" for term, pattern in patterns if pattern.search(flat)]

    assert not violations, "названия технологий из датасета попали в репозиторий:\n" + "\n".join(violations)

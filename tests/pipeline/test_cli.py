"""Тесты консольного вывода: отчёт по-русски не должен падать из-за локали."""

import io
import sys
from pathlib import Path

import pytest

from src.pipeline import cli


def test_report_survives_non_utf8_locale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`cli ... > файл` на локали POSIX/C: раньше печать падала с UnicodeEncodeError."""
    out = tmp_path / "out.txt"
    with out.open("wb") as raw:
        stream = io.TextIOWrapper(raw, encoding="ascii")
        monkeypatch.setattr(sys, "stdout", stream)
        with pytest.raises(UnicodeEncodeError):
            print("натрий-ионные аккумуляторы")
            stream.flush()
        cli._force_utf8_output()
        print("натрий-ионные аккумуляторы")
        stream.flush()
    assert "натрий-ионные аккумуляторы" in out.read_text(encoding="utf-8")

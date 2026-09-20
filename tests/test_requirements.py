"""Зависимости образа не должны разъезжаться с requirements.txt.

В образ ставится подмножество (docker/requirements-app.txt): обучение и отчёты в контейнере не нужны,
а torch с соседями добавляет к образу гигабайты. Подмножество легко забыть обновить — отсюда тест.
"""

from pathlib import Path

FULL = Path("requirements.txt")
APP = Path("docker/requirements-app.txt")

# Пакеты, которых в образе нет намеренно: нужны только на машине разработчика (обучение модели,
# отчёты, тесты и линтер). В контейнере идёт только прогон и выдача.
DEV_ONLY = {
    "scikit-learn",
    "catboost-dev",
    "shap",
    "sentence-transformers",
    "matplotlib",
    "openpyxl",
    "pytest",
    "pytest-asyncio",
    "ruff",
}


def _requirements(path: Path) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    specs = [line for line in lines if line and not line.startswith("#")]
    return {_name(spec): spec for spec in specs}


def _name(spec: str) -> str:
    return spec.split(">=")[0].split("==")[0].split("[")[0].strip()


def test_app_requirements_are_a_subset_with_same_versions():
    full, app = _requirements(FULL), _requirements(APP)
    missing = sorted(set(app) - set(full))
    assert not missing, f"в requirements.txt нет: {missing} — образ поставит то, чего нет в проекте"
    different = {name: (app[name], full[name]) for name in app if app[name] != full[name]}
    assert not different, f"версии разошлись: {different}"


def test_runtime_packages_are_not_forgotten_in_the_image():
    """Всё, что не помечено как нужное только разработчику, должно попасть в образ."""
    full, app = _requirements(FULL), _requirements(APP)
    forgotten = sorted(name for name in full if name not in app and name not in DEV_ONLY)
    assert not forgotten, f"нет в образе: {forgotten} — добавь в docker/requirements-app.txt или в DEV_ONLY"

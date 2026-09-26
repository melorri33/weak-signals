"""Сравнение моделей: разбор списка моделей, возобновление, пропуск модели без ключа, таблица — офлайн."""

from pathlib import Path

import pytest

from src.common.schemas import SearchResult
from src.llm.errors import LLMError
from src.pipeline import compare_models as cm

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "search_result_example.json"


def _example() -> SearchResult:
    return SearchResult.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def test_provider_is_before_the_first_colon():
    assert cm.parse_spec("ollama:qwen3:4b") == ("ollama", "qwen3:4b")
    assert cm.parse_spec("gigachat:GigaChat-2") == ("gigachat", "GigaChat-2")
    with pytest.raises(ValueError, match="провайдер:модель"):
        cm.parse_spec("qwen3")


def test_run_dir_has_no_colons(tmp_path: Path):
    assert cm.run_dir("ollama", "qwen3:4b", tmp_path).name == "ollama__qwen3-4b"


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Конвейер без сети: запоминает запросы и отдаёт пример выдачи."""
    seen: list[str] = []

    async def fake_run(query: str) -> SearchResult:
        seen.append(query)
        return _example().model_copy(update={"query": query})

    monkeypatch.setattr("src.pipeline.run.run", fake_run)
    monkeypatch.setattr("src.llm.providers.make_backend", lambda provider, model: None)
    return seen


async def test_finished_domain_is_not_run_again(tmp_path: Path, fake_pipeline: list[str]):
    domains = {"Финтех": "запрос 1", "Роботы": "запрос 2"}
    out = cm.run_dir("ollama", "qwen3:4b", tmp_path)
    out.mkdir(parents=True)
    (out / "Финтех.json").write_text(_example().model_dump_json(), encoding="utf-8")
    (out / "Роботы.json").write_text(_example().model_copy(update={"status": "error"}).model_dump_json())

    assert await cm.run_model("ollama", "qwen3:4b", domains, tmp_path) == 1
    assert fake_pipeline == ["запрос 2"]  # упавший с ошибкой прогон считается заново
    assert (out / "Роботы.json").exists()
    assert cm.load_rows(tmp_path)[0].model == "ollama:qwen3:4b"  # в отчёте — настоящее имя модели


async def test_model_without_key_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def no_key(provider, model):
        raise LLMError("нужен GIGACHAT_CREDENTIALS")

    monkeypatch.setattr("src.llm.providers.make_backend", no_key)
    assert await cm.run_model("gigachat", "GigaChat-2", {"Финтех": "запрос"}, tmp_path) == 0
    assert not any(tmp_path.iterdir())


def test_report_counts_top_and_llm_calls(tmp_path: Path):
    out = cm.run_dir("gigachat", "GigaChat-2", tmp_path)
    out.mkdir(parents=True)
    result = _example()
    (out / "Финтех.json").write_text(result.model_dump_json(), encoding="utf-8")

    [row] = cm.load_rows(tmp_path)
    assert (row.model, row.domain) == ("gigachat:GigaChat-2", "Финтех")
    assert row.top == len(result.top) and row.documents == result.documents_processed
    assert row.llm_calls == sum(1 for c in result.model_calls if c.step not in cm.NOT_LLM_STEPS)

    report = cm.report_md([row])
    assert "| gigachat:GigaChat-2 | 1 | — |" in report  # без датасета «совпало» пустое, а не ноль
    assert "нет data/*.xlsx" in report


def test_report_lies_next_to_its_runs_dir(tmp_path: Path):
    assert cm.report_path(cm.RUNS_DIR) == cm.REPORT_PATH
    assert cm.report_path(tmp_path / "model_runs_pass2") == tmp_path / "model_runs_pass2.md"

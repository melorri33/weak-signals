"""Офлайн-тесты API: прогон запускается в фоне, его состояние видно по run_id, ошибки не роняют сервер."""

import asyncio
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import evidence, main, saved_runs
from src.api.runs import RunRegistry
from src.common.schemas import Candidate, CandidateFeatures, SearchResult, TermStats

EXAMPLE = Path("tests/fixtures/search_result_example.json")
POLL_LIMIT = 200  # запросов: фальшивый конвейер укладывается в единицы, ограничение — от зависания теста
HOLD_TICK_S = 0.01


def _example_result(run_id: str, query: str) -> SearchResult:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["run_id"] = run_id
    raw["query"] = query
    return SearchResult.model_validate(raw)


@pytest.fixture(autouse=True)
def runs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Готовые прогоны — во временной папке: фальшивые выдачи не должны попасть в настоящую data/search_runs."""
    path = tmp_path / "search_runs"
    monkeypatch.setattr(saved_runs, "RUNS_DIR", path)
    monkeypatch.setattr(evidence, "EVIDENCE_DIR", tmp_path / "search_evidence")
    return path


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch: pytest.MonkeyPatch):
    """Свой реестр на каждый тест и никакой базы: тесты идут офлайн и не зависят друг от друга."""
    monkeypatch.setattr(main, "registry", RunRegistry())
    monkeypatch.setattr("src.pipeline.deps.get_search_result", lambda run_id: None)
    monkeypatch.setattr("src.pipeline.deps.database_ok", lambda: False)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Клиент через `with`: так у приложения один цикл событий на все запросы и фоновый прогон живёт между ними
    (без `with` каждый запрос получает свой цикл и дожидается прогона внутри POST — фон превращается в ожидание).
    """
    with TestClient(main.app) as client:
        yield client


def _fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    steps: int = 2,
    fail: bool = False,
    release: threading.Event | None = None,
) -> None:
    """Конвейер-двойник. release — когда нужно подержать прогон запущенным и отпустить его из теста."""

    async def fake_run(query: str, *, run_id=None, on_progress=None, on_features=None) -> SearchResult:
        result = SearchResult(run_id=run_id, query=query, stage="начинаем")
        if on_features is not None:
            on_features(
                [Candidate(id="zk-kyc", name="zero-knowledge kyc", name_ru="KYC с нулевым разглашением")],
                [CandidateFeatures(candidate_id="zk-kyc", total_pubs=42, growth_3y=0.8)],
                {"zk-kyc": TermStats(term="zero-knowledge kyc", pubs_by_year={2024: 10, 2025: 32})},
            )
        for step in range(steps):
            result.stage = f"шаг {step + 1}"
            if on_progress is not None:
                on_progress(result)
            await asyncio.sleep(0)
        # threading.Event, а не asyncio.Event: тест живёт в другом потоке, чем цикл событий сервера.
        while release is not None and not release.is_set():
            await asyncio.sleep(HOLD_TICK_S)
        if fail:
            raise RuntimeError("источник не ответил")
        return _example_result(run_id, query)

    monkeypatch.setattr("src.api.runs.run_pipeline", fake_run)


def _wait_done(client: TestClient, run_id: str) -> dict:
    """Опрашивать GET /search/{run_id}, пока прогон не закончится, — так же делает интерфейс."""
    for _ in range(POLL_LIMIT):
        body = client.get(f"/search/{run_id}").json()
        if body["status"] != "running":
            return body
    raise AssertionError(f"прогон {run_id} не закончился за {POLL_LIMIT} запросов")


def test_search_runs_in_background_and_returns_top(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _fake_pipeline(monkeypatch)
    started = client.post("/search", json={"query": "перспективные решения в финтехе"})
    assert started.status_code == 202
    body = started.json()
    assert body["status"] == "running" and body["run_id"]

    done = _wait_done(client, body["run_id"])
    assert done["status"] == "done"
    assert len(done["top"]) == len(json.loads(EXAMPLE.read_text(encoding="utf-8"))["top"])
    assert done["query"] == "перспективные решения в финтехе"


def test_second_run_is_refused_while_first_is_going(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    release = threading.Event()
    _fake_pipeline(monkeypatch, release=release)
    first = client.post("/search", json={"query": "роботы для склада"}).json()
    second = client.post("/search", json={"query": "edge вычисления"})
    assert second.status_code == 429
    assert first["run_id"] in second.json()["detail"]
    release.set()
    _wait_done(client, first["run_id"])
    # Прогон закончился — место свободно.
    assert client.post("/search", json={"query": "edge вычисления"}).status_code == 202


def test_failed_pipeline_becomes_status_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _fake_pipeline(monkeypatch, fail=True)
    run_id = client.post("/search", json={"query": "защита ИИ"}).json()["run_id"]
    done = _wait_done(client, run_id)
    assert done["status"] == "error"
    assert "источник не ответил" in done["error"]


def test_unknown_run_is_404(client: TestClient):
    assert client.get("/search/нет-такого").status_code == 404


def test_run_is_taken_from_storage_when_not_in_memory(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    saved = _example_result("saved-run", "индустриальный ИИ")
    monkeypatch.setattr("src.pipeline.deps.get_search_result", lambda run_id: saved if run_id == "saved-run" else None)
    body = client.get("/search/saved-run").json()
    assert body["query"] == "индустриальный ИИ"


def test_signal_card_by_candidate_id(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _fake_pipeline(monkeypatch)
    run_id = client.post("/search", json={"query": "инфраструктура ИИ"}).json()["run_id"]
    done = _wait_done(client, run_id)
    candidate_id = done["top"][0]["candidate_id"]

    card = client.get(f"/signal/{run_id}/{candidate_id}")
    assert card.status_code == 200
    assert card.json()["candidate_id"] == candidate_id
    assert card.json()["sources"], "карточка без источников нарушает ТЗ"
    assert client.get(f"/signal/{run_id}/нет-такого").status_code == 404


def test_short_query_is_rejected(client: TestClient):
    assert client.post("/search", json={"query": "ИИ"}).status_code == 422


def test_health_reports_what_is_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def unavailable(self) -> bool:
        return False

    monkeypatch.setattr("src.llm.client.LLMClient.is_available", unavailable)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["llm_available"] is False and body["database_available"] is False
    assert len(body["notes"]) == 2, "и про модель, и про базу должно быть сказано по-русски"


def test_finished_run_is_saved_to_file_and_survives_restart(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, runs_dir: Path
):
    _fake_pipeline(monkeypatch)
    run_id = client.post("/search", json={"query": "роботы для склада"}).json()["run_id"]
    _wait_done(client, run_id)
    assert (runs_dir / f"{run_id}.json").is_file()

    # Перезапуск сервера: память пустая, базы нет — прогон открывается из файла.
    monkeypatch.setattr(main, "registry", RunRegistry())
    body = client.get(f"/search/{run_id}").json()
    assert body["status"] == "done" and body["query"] == "роботы для склада"


def test_failed_run_is_not_saved_to_file(client: TestClient, monkeypatch: pytest.MonkeyPatch, runs_dir: Path):
    _fake_pipeline(monkeypatch, fail=True)
    run_id = client.post("/search", json={"query": "защита ИИ"}).json()["run_id"]
    _wait_done(client, run_id)
    assert not list(runs_dir.glob("*.json")), "в готовые прогоны попадает только выдача со status=done"


def test_run_is_found_by_content_when_file_is_named_otherwise(client: TestClient, runs_dir: Path):
    """Проверка «как у жюри» пишет прогоны под именем области, а не run_id."""
    runs_dir.mkdir(parents=True)
    saved = _example_result("domain-run", "финтех")
    (runs_dir / "Финтех.json").write_text(saved.model_dump_json(), encoding="utf-8")
    assert client.get("/search/domain-run").json()["query"] == "финтех"


def test_runs_lists_memory_and_files_newest_first(client: TestClient, monkeypatch: pytest.MonkeyPatch, runs_dir: Path):
    runs_dir.mkdir(parents=True)
    old = _example_result("old-run", "старый запрос")
    (runs_dir / "old-run.json").write_text(old.model_dump_json(), encoding="utf-8")
    (runs_dir / "broken.json").write_text("{не json", encoding="utf-8")

    release = threading.Event()
    _fake_pipeline(monkeypatch, release=release)
    running = client.post("/search", json={"query": "идущий прогон"}).json()["run_id"]

    runs = client.get("/runs").json()
    assert [r["run_id"] for r in runs] == [running, "old-run"], "битый файл пропускается, свежий прогон первым"
    assert runs[0]["status"] == "running" and runs[0]["stage"]
    assert runs[1]["signals"] == len(old.top) and runs[1]["confident_signals"] == old.confident_signals
    release.set()
    _wait_done(client, running)


def test_evidence_for_the_signal_map_survives_restart(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Признаки кандидатов отдаются отдельно от выдачи и переживают перезапуск сервера."""
    _fake_pipeline(monkeypatch)
    run_id = client.post("/search", json={"query": "финтех"}).json()["run_id"]
    _wait_done(client, run_id)

    monkeypatch.setattr(main, "registry", RunRegistry())  # «перезапуск»: память пуста, остался файл
    body = client.get(f"/search/{run_id}/evidence").json()

    item = body["candidates"][0]
    assert item["name_ru"] == "KYC с нулевым разглашением"
    assert item["features"]["total_pubs"] == 42
    assert item["pubs_by_year"] == {"2024": 10, "2025": 32}


def test_evidence_of_an_old_run_is_404(client: TestClient):
    assert client.get("/search/abc123/evidence").status_code == 404
    assert client.get("/search/..%2F..%2Fsecret/evidence").status_code == 404

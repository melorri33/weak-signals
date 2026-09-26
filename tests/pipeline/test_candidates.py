"""Офлайн-тесты выделения кандидатов: ответ LLM разбирается и проверяется, отказ не валит шаг."""

import json

import pytest
from pydantic import BaseModel

from src.common.schemas import Candidate, Document, SourceType
from src.llm.client import LLMError
from src.pipeline import candidates as candidates_module
from src.pipeline.candidates import _is_usable, _unslug, extract_candidates, warn_on_long_names


def _doc(doc_id: str, title: str, source_type: SourceType = SourceType.PAPER) -> Document:
    return Document(
        id=doc_id,
        source="openalex",
        source_type=source_type,
        title=title,
        url=f"https://example.org/{doc_id}",
    )


class _FakeClient:
    """Отдаёт заранее заготовленные ответы по очереди и запоминает промпты."""

    def __init__(self, answers: list[dict | Exception]):
        self._answers = list(answers)
        self.prompts: list[str] = []

    async def ask_json(self, step: str, prompt: str, schema: type[BaseModel], max_tokens: int = 0) -> BaseModel:
        self.prompts.append(prompt)
        answer = self._answers.pop(0) if self._answers else {"candidates": []}
        if isinstance(answer, Exception):
            raise answer
        return schema.model_validate_json(json.dumps(answer))


async def test_llm_names_the_technology_not_the_headline():
    """Главное отличие от заглушки: кандидат — это технология, а не кусок заголовка новости."""
    docs = [
        _doc("a", "Startup raises $12M to run language models on phones", SourceType.NEWS),
        _doc("b", "On-device inference for small language models"),
    ]
    client = _FakeClient([{"candidates": [{"name": "on-device inference", "document_ids": ["d1", "d2"]}]}])

    candidates = await extract_candidates(docs, client=client)

    assert [c.name for c in candidates] == ["on-device inference"]
    assert candidates[0].id == "on-device-inference"
    assert sorted(candidates[0].document_ids) == ["a", "b"]
    # Русское название на этом шаге не спрашиваем: его пишет make_card для пятнадцати карточек,
    # а не для полутора сотен кандидатов. См. docstring _Candidate.
    assert candidates[0].name_ru is None


async def test_same_technology_from_two_batches_is_merged():
    docs = [_doc(str(i), f"Работа {i}") for i in range(14)]  # больше одной пачки
    answer = {"candidates": [{"name": "quantum sensing", "document_ids": ["d1"]}]}
    client = _FakeClient([answer, answer])

    candidates = await extract_candidates(docs, client=client)

    assert len(candidates) == 1
    assert len(candidates[0].document_ids) == 2  # по одному документу из каждой пачки
    assert len(client.prompts) == 2


async def test_invented_document_ids_are_dropped():
    """Модель сослалась на документ, которого в пачке не было: карточку по нему не собрать — выкидываем."""
    docs = [_doc("a", "Работа")]
    client = _FakeClient(
        [
            {
                "candidates": [
                    {"name": "invented technology", "document_ids": ["d99"]},
                    {"name": "quantum sensing", "document_ids": ["d1"]},
                ]
            }
        ]
    )

    candidates = await extract_candidates(docs, client=client)

    assert [c.name for c in candidates] == ["quantum sensing"]


async def test_too_broad_names_are_dropped():
    docs = [_doc("a", "Работа")]
    client = _FakeClient(
        [
            {
                "candidates": [
                    {"name": "artificial intelligence", "document_ids": ["d1"]},
                    {"name": "on-device inference", "document_ids": ["d1"]},
                ]
            }
        ]
    )

    candidates = await extract_candidates(docs, client=client)

    assert [c.name for c in candidates] == ["on-device inference"]


async def test_news_go_to_the_model_first():
    """71% источников датасета — техноновости, поэтому в пачку они должны попадать раньше науки."""
    docs = [_doc(str(i), f"Статья {i}") for i in range(12)]
    docs.append(_doc("news", "Новость про стартап", SourceType.NEWS))
    client = _FakeClient([{"candidates": [{"name": "quantum sensing", "document_ids": ["d1"]}]}])

    await extract_candidates(docs, client=client)

    assert "Новость про стартап" in client.prompts[0]


async def test_falls_back_to_titles_when_llm_is_down():
    """LLM недоступна — шаг не падает: названия режутся из заголовков, прогон продолжается."""
    docs = [
        _doc("a", "Zero-knowledge KYC verification for cross-border payments"),
        _doc("b", "Zero-knowledge KYC verification: benchmarks"),
    ]
    client = _FakeClient([LLMError("Ollama не ответила")])

    candidates = await extract_candidates(docs, client=client)

    assert [c.id for c in candidates] == ["zero-knowledge-kyc-verification"]
    assert candidates[0].document_ids == ["a", "b"]


async def test_no_documents_no_candidates():
    assert await extract_candidates([]) == []


def test_long_name_is_reported(caplog: pytest.LogCaptureFixture):
    """Длинное название: статистику точной фразой по нему не найти — предупреждаем, но кандидата не теряем."""
    candidates = [
        Candidate(id="onboarding", name="Защита онбординга от инъекционных атак и дипфейков"),
        Candidate(id="na-ion", name="sodium-ion batteries"),
    ]

    with caplog.at_level("WARNING", logger="src.pipeline.candidates"):
        returned = warn_on_long_names(candidates)

    assert returned == candidates, "кандидатов не выбрасываем, только предупреждаем"
    assert "Названия длиннее" in caplog.text
    assert "Защита онбординга" in caplog.text
    assert "sodium-ion batteries" not in caplog.text


def test_short_names_do_not_warn(caplog: pytest.LogCaptureFixture):
    with caplog.at_level("WARNING", logger="src.pipeline.candidates"):
        warn_on_long_names([Candidate(id="na-ion", name="sodium-ion batteries")])

    assert caplog.text == ""


async def test_all_answers_unusable_falls_back_to_titles():
    """Модель ответила, но брать нечего — лучше грубые названия из заголовков, чем пустой шаг."""
    docs = [_doc("a", "Sodium-ion batteries for grid storage")]
    client = _FakeClient([{"candidates": [{"name": "blockchain", "document_ids": ["d1"]}]}])

    candidates = await extract_candidates(docs, client=client)

    assert [c.name for c in candidates] == ["Sodium-ion batteries grid"]


async def test_stops_itself_before_the_budget_ends(monkeypatch: pytest.MonkeyPatch):
    """Отмена снаружи приходит посреди вызова модели и уносит всех выписанных кандидатов,
    поэтому шаг останавливается сам и отдаёт то, что успел."""
    import asyncio

    from src.pipeline import candidates as module

    monkeypatch.setattr(module, "BUDGET_S", 0.3)
    monkeypatch.setattr(module, "DOCS_IN_BATCH", 1)
    docs = [_doc(str(i), f"Работа {i}") for i in range(5)]

    class _SlowClient(_FakeClient):
        async def ask_json(self, step, prompt, schema, max_tokens=0):
            await asyncio.sleep(0.2)
            return await super().ask_json(step, prompt, schema, max_tokens)

    client = _SlowClient([{"candidates": [{"name": f"технология {i}", "document_ids": ["d1"]}]} for i in range(5)])

    result = await extract_candidates(docs, client=client)

    assert result, "то, что успели выписать, должно вернуться"
    assert len(client.prompts) < 5, "последние пачки не берём — они не успеют закончиться"


def test_order_mixes_news_and_science():
    """Наука должна попадать в очередь к модели, а не ждать за всеми новостями.

    Шаг успевает прочитать около сотни документов из тысячи с лишним. При строгой очереди
    «сначала новости» наука не доходила до модели никогда, хотя arXiv находит технологий
    датасета больше, чем новостные ленты (замер 21.09: 69 против 63 из 100).

    Проверяем именно начало очереди: то, что за пределами первой сотни, модель не увидит.
    """
    docs = [
        Document(
            id=f"n{i}",
            source="techcrunch.com",
            source_type=SourceType.NEWS,
            title=f"Новость {i}",
            url=f"https://example.com/n{i}",
        )
        for i in range(20)
    ] + [
        Document(
            id=f"p{i}",
            source="arxiv",
            source_type=SourceType.PREPRINT,
            title=f"Препринт {i}",
            url=f"https://example.org/p{i}",
        )
        for i in range(20)
    ]

    order = candidates_module._order_for_llm(docs)

    assert len(order) == len(docs), "ни один документ не теряется"
    first_twelve = [d.source_type for d in order[:12]]
    assert SourceType.NEWS in first_twelve and SourceType.PREPRINT in first_twelve, (
        "в первой дюжине — то, что реально успеет прочитать модель, — должны быть оба вида"
    )
    assert first_twelve.count(SourceType.NEWS) == first_twelve.count(SourceType.PREPRINT), (
        "новости и наука делят очередь поровну: замер 21.09 на корпусе из шести областей дал "
        "5 технологий датасета из 12 при чередовании один к одному, 4 при двух новостях на статью "
        "и 2 при строгой очереди «сначала новости»"
    )


async def test_description_instead_of_a_term_is_dropped():
    """Название длиннее четырёх слов — это описание, а не термин.

    Замер 21.09 по прогону из шести областей: у названий в 2-3 слова статистика публикаций
    находится в 70-94% случаев, у пятисловных и длиннее — в 14%, средняя оценка 0.06,
    и ни одно не попало в топ-15. По фразе, которой никто не пишет, публикаций не найти,
    и такой кандидат ранжируется вслепую, занимая место в списке.
    """
    docs = [_doc("a", "Startup raises $12M to run language models on phones", SourceType.NEWS)]
    client = _FakeClient(
        [
            {
                "candidates": [
                    {"name": "identity management for large language model agents", "document_ids": ["d1"]},
                    {"name": "on-device inference", "document_ids": ["d1"]},
                ]
            }
        ]
    )

    candidates = await extract_candidates(docs, client=client)

    assert [c.name for c in candidates] == ["on-device inference"]


def test_every_plural_generic_head_has_its_singular():
    """Для каждого слова во множественном числе должно быть единственное.

    Именно такого пропуска стоила ошибка: «platforms» в списке было, «platform» не было,
    и записи вроде «… platform» проходили фильтр и занимали места в топ-15. Замер 21.09
    по двум прогонам: 5 и 8 таких записей из 90 мест выдачи.

    Обратную сторону не проверяем: у «processing» и «infrastructure» естественного
    множественного числа нет, и требовать его бессмысленно.
    """
    from src.pipeline.candidates import _TOO_GENERIC_HEAD

    missing = []
    for word in sorted(_TOO_GENERIC_HEAD):
        if word.endswith("ies"):
            singular = f"{word[:-3]}y"
        elif word.endswith("s") and not word.endswith("ss"):
            singular = word[:-1]
        else:
            continue
        if singular not in _TOO_GENERIC_HEAD:
            missing.append(f"«{word}» есть, «{singular}» нет")
    assert not missing, "в списке общих слов не хватает форм: " + "; ".join(missing)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("robotic-process-automation", "robotic process automation"),
        ("sodium-ion battery", "sodium-ion battery"),
        ("humanoid-robot", "humanoid-robot"),
        ("post-quantum-cryptography", "post quantum cryptography"),
    ],
)
def test_slug_names_become_words(raw: str, expected: str):
    """GigaChat пишет названия слагами: делаем из них слова, настоящие дефисы в терминах не трогаем."""
    assert _unslug(raw) == expected


def test_slug_with_generic_head_is_dropped_after_unslug():
    """«sensat-data-platform» проходил как одно слово; после разбора срабатывает проверка общего слова."""
    assert not _is_usable(_unslug("sensat-data-platform"))


async def test_cap_takes_candidates_from_every_batch(monkeypatch: pytest.MonkeyPatch):
    """Предел не должен срезать всё найденное в последних пачках: берём по кругу."""
    monkeypatch.setattr(candidates_module, "MAX_CANDIDATES", 4)
    docs = [_doc(str(i), f"Работа {i}") for i in range(16)]  # две пачки по 8
    first = {"candidates": [{"name": f"early sensor {n}", "document_ids": ["d1"]} for n in "abcd"]}
    second = {"candidates": [{"name": "late sensor", "document_ids": ["d1"]}]}
    client = _FakeClient([first, second])

    names = [c.name for c in await extract_candidates(docs, client=client)]

    assert "late sensor" in names
    assert names[:2] == ["early sensor a", "late sensor"]


@pytest.mark.parametrize(
    "variants",
    [
        ["algae bioreactors", "algae-bioreactors", "algae bioreactor"],
        ["solid-state batteries", "solid state battery"],
        ["heat pump compressors", "heat pump compressor"],
    ],
)
async def test_plural_and_hyphen_variants_are_one_candidate(variants: list[str]):
    docs = [_doc(str(i), f"Работа {i}") for i in range(8)]
    answer = {"candidates": [{"name": v, "document_ids": [f"d{n + 1}"]} for n, v in enumerate(variants)]}

    candidates = await extract_candidates(docs, client=_FakeClient([answer]))

    assert len(candidates) == 1 and candidates[0].name == variants[0]
    assert len(candidates[0].document_ids) == len(variants)


@pytest.mark.parametrize("word", ["robotics", "analysis", "glass", "consensus"])
def test_singular_keeps_words_that_only_look_plural(word: str):
    assert candidates_module._singular(word) == word


def _news(title: str, abstract: str) -> Document:
    return Document(
        id="n", source="news", source_type=SourceType.NEWS, title=title, abstract=abstract, url="https://e.x/n"
    )


@pytest.mark.parametrize(
    ("name", "title", "abstract"),
    [
        ("harvestiq", "HarvestIQ closes $12M round", "The startup HarvestIQ makes crop yield models."),
        ("bayasystems", "Baya Systems raises $36M to propel chiplet innovation", "Baya Systems builds fabrics."),
        ("mistral ai model", "Cloudera brings Mistral AI models on premises", "Cloudera will offer Mistral AI models."),
        ("alexandre lebrun", "Alexandre Lebrun leaves the company", "Alexandre Lebrun, a founder, said."),
        # Имя с опечаткой.
        ("havrestiq", "HarvestIQ closes $12M round", "The startup HarvestIQ makes crop yield models."),
    ],
)
def test_company_product_and_person_names_are_not_technologies(name: str, title: str, abstract: str):
    assert candidates_module._is_proper_name(name, [_news(title, abstract)])


@pytest.mark.parametrize(
    ("name", "title", "abstract"),
    [
        ("neocloud", "Groq raises $350M to fuel its pivot from AI chips to neocloud", "The neocloud market grows."),
        # Заголовок «всё с заглавной» — не признак имени.
        ("solid state hydrogen storage", "Startup Secures $80M to Scale Solid State Hydrogen Storage", ""),
        ("chiplet interconnect", "Baya Systems raises $36M", "Baya Systems builds chiplet interconnect fabrics."),
        # «Quantum» из «IBM Quantum» — одиночное слово с заглавной, не имя.
        ("quantum computing", "IBM Quantum hits milestone", "IBM Quantum said quantum computing will scale."),
        ("SASE", "Firms adopt SASE", "SASE grows."),
        # Однословная технология, не похожая на имена в тексте.
        ("chiplets", "Baya Systems raises $36M to propel AI and chiplet innovation", "Baya Systems builds fabrics."),
    ],
)
def test_technologies_are_kept(name: str, title: str, abstract: str):
    assert not candidates_module._is_proper_name(name, [_news(title, abstract)])


async def test_company_name_does_not_take_a_candidate_slot():
    docs = [_news("HarvestIQ closes round for its yield models", "HarvestIQ builds crop yield forecasting.")]
    answer = {
        "candidates": [
            {"name": "harvestiq", "document_ids": ["d1"]},
            {"name": "crop yield forecasting", "document_ids": ["d1"]},
        ]
    }
    names = [c.name for c in await extract_candidates(docs, client=_FakeClient([answer]))]
    assert names == ["crop yield forecasting"]


@pytest.mark.parametrize("name", ["gpt-6", "stretch-4", "gemma_3", "artificial-intelligence", "agentic_ai"])
async def test_versions_and_broad_terms_in_any_spelling_are_dropped(name: str):
    docs = [_doc("a", "Работа")]
    answer = {
        "candidates": [{"name": name, "document_ids": ["d1"]}, {"name": "quantum sensing", "document_ids": ["d1"]}]
    }
    names = [c.name for c in await extract_candidates(docs, client=_FakeClient([answer]))]
    assert names == ["quantum sensing"]


def test_possessive_does_not_hide_a_company_name():
    doc = _news("Cloudera brings models on premises", "Cloudera will offer Mistral AI’s frontier models.")
    assert candidates_module._is_proper_name("mistral ai model", [doc])


async def test_name_that_is_the_company_itself_is_dropped():
    docs = [_doc("a", "Работа")]
    answer = {
        "candidates": [
            {"name": "acme agro sensors", "company": "Acme Agro", "document_ids": ["d1"]},
            {"name": "soil moisture sensing", "company": "Acme Agro", "document_ids": ["d1"]},
        ]
    }
    names = [c.name for c in await extract_candidates(docs, client=_FakeClient([answer]))]
    assert names == ["soil moisture sensing"]

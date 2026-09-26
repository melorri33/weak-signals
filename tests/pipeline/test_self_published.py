"""Самопубликации без рецензии убираются из сбора — офлайн."""

import pytest

from src.common.schemas import Document, SourceType
from src.pipeline import deps
from src.pipeline.self_published import is_self_published, without_self_published


def _doc(url: str, doc_id: str = "openalex:W1") -> Document:
    return Document(id=doc_id, source="openalex", source_type=SourceType.PAPER, title="Статья", url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://doi.org/10.5281/zenodo.123456",
        "https://doi.org/10.13140/RG.2.2.12345.67890",
        "https://doi.org/10.6084/m9.figshare.123",
        "https://zenodo.org/records/123456",
        "https://www.researchgate.net/publication/123",
    ],
)
def test_self_published_repositories(url: str):
    assert is_self_published(_doc(url))


@pytest.mark.parametrize(
    "url",
    [
        "https://doi.org/10.1038/s41586-024-00001-1",  # Nature
        "https://doi.org/10.2139/ssrn.1234567",  # SSRN — препринты с модерацией, оставляем
        "https://arxiv.org/abs/2601.00001",
        "https://techcrunch.com/2026/06/20/zenodo-raises/",
    ],
)
def test_reviewed_and_news_are_kept(url: str):
    assert not is_self_published(_doc(url))


def test_filter_keeps_order_of_the_rest():
    docs = [
        _doc("https://doi.org/10.1038/a", "openalex:W1"),
        _doc("https://doi.org/10.5281/zenodo.1", "openalex:W2"),
        _doc("https://arxiv.org/abs/1", "arxiv:1"),
    ]
    assert [d.id for d in without_self_published(docs)] == ["openalex:W1", "arxiv:1"]


async def test_collect_drops_them_for_both_rounds(monkeypatch: pytest.MonkeyPatch):
    async def real_collect(phrases, limit):
        return [_doc("https://doi.org/10.5281/zenodo.1", "openalex:W2"), _doc("https://arxiv.org/abs/1", "arxiv:1")]

    monkeypatch.setattr(deps, "_load", lambda path, name: real_collect)
    assert [d.id for d in await deps.collect(["фраза"], limit=10)] == ["arxiv:1"]

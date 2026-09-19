"""Склейка документов по DOI и по нормализованному заголовку."""

from __future__ import annotations

from src.collectors.dedupe import dedupe
from src.common.schemas import Document, SourceType


def _doc(**kwargs) -> Document:
    defaults = {
        "id": "openalex:W1",
        "source": "openalex",
        "source_type": SourceType.PAPER,
        "title": "Zero-knowledge KYC",
        "url": "https://doi.org/10.1/abc",
    }
    return Document(**{**defaults, **kwargs})


def test_same_doi_from_different_sources_is_one_document():
    a = _doc(id="openalex:W1", source="openalex", url="https://doi.org/10.1/abc")
    b = _doc(id="arxiv:1234", source="arxiv", url="http://dx.doi.org/10.1/ABC")  # тот же DOI, другой регистр/протокол

    result = dedupe([a, b])

    assert len(result) == 1
    assert result[0].id == "openalex:W1"  # первый источник побеждает


def test_same_title_without_doi_is_one_document():
    a = _doc(id="openalex:W1", url="https://example.org/1", title="Zero-Knowledge KYC!")
    b = _doc(id="arxiv:2", url="https://example.org/2", title="zero knowledge kyc")

    result = dedupe([a, b])

    assert len(result) == 1


def test_different_documents_are_kept():
    a = _doc(id="openalex:W1", url="https://example.org/1", title="Zero-knowledge KYC")
    b = _doc(id="openalex:W2", url="https://example.org/2", title="Federated credit scoring")

    result = dedupe([a, b])

    assert len(result) == 2

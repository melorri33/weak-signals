"""Уровни доверия на реальных URL (в том числе из типов источников датасета организаторов)."""

import pytest

from src.common.schemas import SourceType, TrustLevel
from src.trust import level_for
from src.trust.levels import domain_level

H, M, L = TrustLevel.HIGH, TrustLevel.MEDIUM, TrustLevel.LOW
S = SourceType


@pytest.mark.parametrize(
    ("url", "source_type", "expected"),
    [
        # наука, патенты, регуляторы, стандарты — высокий
        ("https://www.nature.com/articles/s41586-025-01234-5", S.PAPER, H),
        ("https://doi.org/10.1145/3544548.3580000", S.PAPER, H),
        ("https://ieeexplore.ieee.org/document/10000000", S.PAPER, H),
        ("https://patents.google.com/patent/US11500000B2", S.PATENT, H),
        ("https://www.cbr.ru/fintech/", S.GOV, H),
        ("https://www.bis.org/publ/bppdf/bispap150.htm", S.REPORT, H),
        ("https://www.nist.gov/pqc", S.GOV, H),
        ("https://www.europol.europa.eu/publications-events", S.REPORT, H),
        ("https://csail.mit.edu/news/robots", S.NEWS, H),
        ("https://owasp.org/www-project-top-10-for-large-language-model-applications/", S.OTHER, H),
        # препринты, отраслевые СМИ, аналитика — средний
        ("https://arxiv.org/abs/2501.00001", S.PREPRINT, M),
        ("https://techcrunch.com/2026/03/01/startup-raises/", S.NEWS, M),
        ("https://siliconangle.com/2026/07/28/ai-model-compression/", S.NEWS, M),
        ("https://www.eetimes.com/analog-design-notes/", S.NEWS, M),
        ("https://www.datacenterdynamics.com/en/news/cooling/", S.NEWS, M),
        ("https://www.reuters.com/technology/ai/", S.NEWS, M),
        ("https://www.rbc.ru/technology_and_media/", S.NEWS, M),
        ("https://github.com/invariantlabs-ai/mcp-scan", S.OTHER, M),
        # пресс-релизы, блоги, соцсети, агрегаторы — низкий
        ("https://www.prnewswire.com/news-releases/company-launches.html", S.PRESS_RELEASE, L),
        ("https://www.businesswire.com/news/home/2026/", S.NEWS, L),
        ("https://medium.com/@someone/ai-agents", S.BLOG, L),
        ("https://habr.com/ru/articles/800000/", S.BLOG, L),
        ("https://x.com/someone/status/1", S.SOCIAL, L),
        ("https://www.linkedin.com/posts/someone", S.SOCIAL, L),
        ("https://news.google.com/rss/articles/abc", S.NEWS, L),
        # неизвестный домен — по типу источника
        ("https://unknown-lab.example/paper.pdf", S.PAPER, H),
        ("https://some-startup.io/blog/launch", S.BLOG, L),
        ("https://small-industry-portal.example/news/1", S.NEWS, M),
        ("https://some-startup.io/press/launch", S.PRESS_RELEASE, L),
    ],
)
def test_level_for(url, source_type, expected):
    assert level_for(url, source_type) == expected


def test_suffix_does_not_match_lookalike_domains():
    assert domain_level("https://notnature.com/a") is None
    assert domain_level("https://nature.com.evil.example/a") is None
    assert domain_level("") is None

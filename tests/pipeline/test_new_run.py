"""Каждый прогон начинается со всеми лентами: «исчерпанные» в прошлом прогоне снова в работе — офлайн."""

from src.collectors import news
from src.pipeline import deps


def test_new_run_forgets_exhausted_feeds():
    news._exhausted.add("siliconangle")
    deps.new_run()
    assert not news._exhausted

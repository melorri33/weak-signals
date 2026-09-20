"""Настройки проекта. Значения берутся из переменных окружения / файла .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # База
    database_url: str = "postgresql+psycopg://weak:weak@localhost:5432/weak_signals"

    # LLM: где живёт модель. ollama — на своей машине, yandexgpt — облако из перечня ТЗ.
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    # Внутри контейнера localhost — это сам контейнер, поэтому адрес Ollama там другой.
    ollama_url_in_docker: str = "http://host.docker.internal:11434"
    # Доступы к облаку Яндекса: сервисный аккаунт с ролью ai.languageModels.user и каталог.
    # Пустые по умолчанию — без них провайдер yandexgpt просто не поднимется, локальный не затронут.
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_api_url: str = "https://ai.api.cloud.yandex.net/v1/chat/completions"
    # Ответы модели кэшируются по тексту запроса: повторный прогон и демонстрация идут мгновенно,
    # а на машине без видеокарты один вызов стоит десятки секунд. Выключается на время замеров скорости.
    llm_cache: bool = True

    # Эмбеддинги
    embed_model: str = "BAAI/bge-m3"

    # Источники
    contact_email: str = ""  # для вежливых запросов к API (User-Agent / mailto)
    openalex_api_key: str = ""  # бесплатный аккаунт openalex.org: $1/день вместо $0.10 без ключа
    patentsview_api_key: str = ""
    lens_api_token: str = ""

    # Бюджеты конвейера
    max_documents: int = 2000
    source_timeout_s: float = 15.0
    collect_budget_s: float = 60.0
    cache_ttl_days: int = 7


@lru_cache
def get_settings() -> Settings:
    return Settings()

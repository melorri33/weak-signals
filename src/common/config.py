"""Настройки проекта. Значения берутся из переменных окружения / файла .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # База
    database_url: str = "postgresql+psycopg://weak:weak@localhost:5432/weak_signals"

    # LLM (локально через Ollama)
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"

    # Эмбеддинги
    embed_model: str = "BAAI/bge-m3"

    # Источники
    contact_email: str = ""  # для вежливых запросов к API (User-Agent / mailto)
    openalex_api_key: str = ""  # бесплатный аккаунт openalex.org: $1/день вместо $0.10 без ключа
    patentsview_api_key: str = ""
    lens_api_token: str = ""

    # Бюджеты конвейера
    max_documents: int = 500
    source_timeout_s: float = 15.0
    collect_budget_s: float = 60.0
    cache_ttl_days: int = 7


@lru_cache
def get_settings() -> Settings:
    return Settings()

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
    # Пауза после вызова локальной модели — доля длительности вызова. 0.25 ≈ 80% загрузки видеокарты
    # в среднем: прогон дольше, зато карта не работает часами на пределе. 0 — без пауз.
    llm_gpu_rest_share: float = 0.0
    # Второй вопрос модели по новостям о стартапах: «какая технология стоит за компанией»
    # (src/pipeline/candidates.py::_rescue). Стоит вызова модели почти на каждую пачку: на локальной
    # модели прогон 27.09 читал на треть меньше документов. Включать на быстрой облачной модели.
    extract_rescue: bool = False
    # Множитель бюджетов шагов с вызовами модели. Облачная модель отвечает на пачку за 3–7 с, но у ключа
    # физлица запрос идёт один за раз, и за обычный бюджет она не дочитывает 500 документов.
    # 2–3 — дочитать все (прогон дольше); 1 — как было.
    llm_budget_scale: float = 1.0
    # Сколько документов из собранных отдаём модели выделения (src/pipeline/candidates.py). Собираем около
    # 2000 на запрос; GigaChat 2 Max читает пачку из 8 за 3–4 с — 1000 документов примерно за 10 минут.
    max_docs_for_llm: int = 500
    # Проверка верха списка моделью: не метод ли из статьи и не зрелый стандарт (src/pipeline/technology.py).
    # Выключается, чтобы сравнить прогоны.
    check_technology: bool = True
    # Поправки к ранжированию по эмбеддингам (src/model/relevance.py): вес близости к теме запроса
    # (0 — выключить) и порог почти-дубля по смыслу названий (1 — выключить).
    topic_relevance_weight: float = 0.6
    near_duplicate_threshold: float = 0.85

    # Эмбеддинги
    embed_model: str = "BAAI/bge-m3"

    # Источники
    contact_email: str = ""  # для вежливых запросов к API (User-Agent / mailto)
    openalex_api_key: str = ""  # бесплатный аккаунт openalex.org: $1/день вместо $0.10 без ключа
    patentsview_api_key: str = ""
    lens_api_token: str = ""

    # Фразы свежести (src/common/phrases.py): к фразам модели добавляется поиск новостей о стартапах
    # области запроса. Выключается, чтобы сравнить прогоны с ними и без них.
    fresh_phrases: bool = True

    # Бюджеты конвейера
    max_documents: int = 2000
    source_timeout_s: float = 15.0
    collect_budget_s: float = 60.0
    cache_ttl_days: int = 7
    # Сколько кредитов OpenAlex нужно на один запрос: меньше — прогон не начинаем, а честно пишем, что лимит
    # исчерпан (замер 28.09: запрос тратит 520–690 кредитов). Иначе прогон шёл бы без статистики публикаций
    # и выдавал испорченную выдачу, похожую на обычную. 0 — не проверять.
    openalex_min_credits: int = 800


@lru_cache
def get_settings() -> Settings:
    return Settings()


def llm_budget(seconds: float) -> float:
    """Бюджет шага с вызовами модели с учётом пауз для видеокарты (LLM_GPU_REST_SHARE).

    Пауза должна удлинять прогон, а не урезать работу. Прогон 27.09 с долей 0.25 и прежними бюджетами:
    шаг выделения прочитал на треть меньше документов, а в двух областях из шести не успели
    собраться карточки — 10 и 13 вместо 15.

    LLM_BUDGET_SCALE — общий множитель поверх паузы: облачной модели — время дочитать все документы.
    """
    s = get_settings()
    return seconds * (1 + max(s.llm_gpu_rest_share, 0.0)) * max(s.llm_budget_scale, 0.1)

# Образ приложения: и API, и интерфейс, и консольный прогон — команда задаётся в docker-compose.yml.
# Модель в образ не кладём: локально она живёт в Ollama (на хосте или отдельным сервисом),
# в облаке — у провайдера из списка ТЗ.
FROM python:3.11-slim

# PYTHONIOENCODING: вывод отчёта по-русски не должен зависеть от локали внутри контейнера.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Зависимости отдельным слоем: правка кода не пересобирает установку пакетов.
# torch — сборка для процессора: модели названий и близости к теме хватает CPU, а сборка с CUDA
# весит на несколько гигабайт больше. Ставим её до requirements-app.txt, иначе sentence-transformers
# притянет сборку с CUDA.
COPY docker/requirements-app.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-app.txt

# Код и всё, что читается на прогоне: правила отсева и доверия (config), обученная модель
# (src/model/artifacts), фикстуры и тестовые запросы (нужны заглушкам и проверке «как у жюри»).
COPY src ./src
COPY config ./config
COPY tests/fixtures ./tests/fixtures
COPY tests/queries.yaml ./tests/
COPY pyproject.toml README.md ./

# Результаты прогонов и кэш ответов модели — на смонтированном томе ./data. Туда же — веса bge-m3
# (около 2 ГБ): качаются при первом прогоне один раз и переживают пересборку образа.
ENV HF_HOME=/app/data/hf
VOLUME /app/data

EXPOSE 8000 8501

# Пользователь остаётся root: data/ приходит с хоста как bind mount, и с непривилегированным
# пользователем на Linux запись в него упирается в чужого владельца папки.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Используем минимальный Python образ
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
WORKDIR /app
COPY requirements.txt .

# Установка torch и sentence-transformers вручную (CPU-only)
RUN pip install --no-cache-dir torch==2.2.2+cpu sentence-transformers==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Установка всех остальных зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Очистка кэша pip и временных файлов
RUN apt-get clean && rm -rf /root/.cache /root/.npm /tmp/* /var/tmp/*

# Копируем всё приложение
COPY . .

# Экспортируем порт
EXPOSE 8000

# Команда запуска FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

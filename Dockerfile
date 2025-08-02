FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
COPY .env .env

RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Указываем рабочий путь к src
ENV PYTHONPATH=/app/src

# Подтягиваем переменные окружения
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV PGVECTOR_URL=${PGVECTOR_URL}

#для FastAPI
EXPOSE 8000

#Газуем
CMD ["python", "run_api.py"]



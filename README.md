# 🧠 PDF Entity Extraction Agent with LLM, RAG and Evaluation

Этот проект реализует pipeline для извлечения сущностей из PDF-документов с использованием:

- 📄 LangChain + OpenAI LLM
- 🔍 RAG с PGVector (PostgreSQL + векторные embeddings)
- ✅ Строгая проверка через Pydantic
- 🔁 Повторы (`retry`) и логирование ошибок
- 📊 Автоматическая оценка качества (precision, recall, f1)
- 🖼️ Подключение Arize Phoenix для мониторинга
- 🌐 FastAPI API-интерфейс для подачи документов

---

## 🚀 Быстрый запуск (с Docker Compose)

1. Клонируй проект и перейди в директорию:

```bash
git clone https://github.com/your/repo.git
cd your-repo
```

2. Создай `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

3. Запусти проект:

```bash
docker-compose up --build
```

---

## 📤 API-доступ (FastAPI)

После запуска будет доступен OpenAPI Swagger UI по адресу:

📍 [http://localhost:8000/docs](http://localhost:8000/docs)

### Примеры эндпоинтов:

#### `POST /upload-pdf`
Загрузить PDF-файл на сервер и получить результат:

```bash
curl -X POST http://localhost:8000/upload-pdf \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@pdfs/contract1.pdf"
```

#### `POST /analyze-text`
Отправить уже извлечённый текст и получить JSON-ответ:

```bash
curl -X POST http://localhost:8000/analyze-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Ваш текст из PDF здесь..."}'
```

#### 📦 Примеры запросов на Python (через `requests`)

```python
import requests

# Отправка текста
text_payload = {"text": "Ваш текст из PDF здесь..."}
response = requests.post("http://localhost:8000/analyze-text", json=text_payload)
print(response.json())

# Загрузка PDF-файла
with open("pdfs/contract1.pdf", "rb") as f:
    files = {"file": ("contract1.pdf", f, "application/pdf")}
    response = requests.post("http://localhost:8000/upload-pdf", files=files)
    print(response.json())
```

---

## 📦 Структура проекта

```bash
src/
├── api.py                    # FastAPI endpoints
├── pdf_llm_agent_pipeline.py # Основной агент
├── retriever_pgvector.py     # Векторное хранилище (PGVector)
├── llm_setup.py              # Настройка LLM
├── models.py                 # Pydantic модели
├── text_extraction.py        # Извлечение текста из PDF
├── phoenix_setup.py          # Подключение Arize Phoenix
├── eval_utils.py             # Метрики и экспорт в Excel
run_batch_evaluation.py       # Оценка качества по Excel
Dockerfile
requirements.txt
docker-compose.yml
.env
```

---

## 🧪 Тестирование и оценка

### 📄 Шаг 1: Подготовка Excel с ground truth

Создай файл `data/test_ground_truth.xlsx` со структурой:

| filename      | inn         | date       | has_stamp | is_offer | mentions_guarantee |
|---------------|-------------|------------|-----------|----------|--------------------|
| contract1.pdf | 7707083893  | 2023-09-15 | TRUE      | FALSE    | TRUE               |
| contract2.pdf | 1234567890  | 2022-11-01 | FALSE     | TRUE     | FALSE              |

> ⚠️ PDF-файлы должны лежать в папке `pdfs/`


### 🧠 Шаг 2: Запуск оценки

```bash
python run_batch_evaluation.py
```

Будет сформирован файл: `results/predictions_with_metrics.xlsx`

### 📊 Метрики:
- Precision
- Recall
- F1-score
- Accuracy

Также выводится сводка по всем документам в консоли.

---

## 📌 Используемые технологии
- OpenAI GPT-4 (или o4-mini)
- LangChain (RAG)
- PGVector + SQLAlchemy
- Pydantic
- FastAPI
- Arize Phoenix
- Docker / docker-compose

---

## ✅ TODO / Возможности расширения
- 🔍 Визуализация ошибок в UI
- 📁 Поддержка zip-архивов с PDF
- 📉 Обучение собственной embedding-модели
- 🧪 Интеграция с CI для автоматической регрессии

---

Если у тебя остались вопросы — открывай Issue или пиши в чат!


Если у тебя остались вопросы — открывай Issue или пиши в чат!


"""
Lва эндпойнта:

POST /upload-pdf — загрузка и обработка PDF

POST /upload-text — обработка уже готового текста

✅ run_agent_on_text() используется в обоих случаях
✅ Парсинг ground_truth из строки
✅ Поддержка запуска через run_api.py

"""

docker-compose up --build
Поднимается FastAPI API + Postgres с pgvector + вся инфраструктура.

И можно сразу тестировать LLM-агента по http://localhost:8000/docs



Агент извлекает сущности, и для каждой мы проверяем True (нашёл) или False (не нашёл):

Термин	Что означает
True Positive (TP)	Агент правильно определил, что сущность есть (True)
False Positive (FP)	Агент сказал, что сущность есть, но её нет
False Negative (FN)	Агент пропустил сущность, хотя она была

📊 Метрики качества извлечения сущностей
Для оценки качества извлечения сущностей агентом используются стандартные метрики классификации:

🔹 Precision (точность)
Из всех сущностей, которые агент отметил как найденные — сколько действительно были верными?

Precision
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑃
Precision= 
TP+FP
TP
​
 
TP (True Positive) — агент правильно предсказал наличие сущности

FP (False Positive) — агент ошибочно указал, что сущность есть, но её нет

🔹 Recall (полнота)
Из всех сущностей, которые должны быть найдены — сколько агент действительно нашёл?

Recall
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑁
Recall= 
TP+FN
TP
​
 
FN (False Negative) — агент пропустил сущность, хотя она была в тексте

🔹 F1-score
Сбалансированная метрика между precision и recall

F1
=
2
×
Precision
×
Recall
Precision
+
Recall
F1=2× 
Precision+Recall
Precision×Recall
​
 
📋 Пример
Предположим, для одного документа:

Сущность	Ожидание (Ground Truth)	Предсказание (Prediction)	Совпадение
has_stamp	True	True	✅ TP
is_offer	False	True	❌ FP
mentions_guarantee	True	False	❌ FN
inn	1234567890	1234567890	✅ TP
date	2023-01-01	2022-01-01	❌ FN

TP = 2

FP = 1

FN = 2

Метрики:

Precision = 2 / (2 + 1) = 0.666

Recall = 2 / (2 + 2) = 0.5

F1 = 2 * (0.666 * 0.5) / (0.666 + 0.5) ≈ 0.571

Эти метрики особенно полезны при тестировании на множестве документов (batch evaluation) с Excel-файлом ground truth.
# 🧠 PDF Entity Extraction Agent with LLM, RAG и Evaluation (локальная модель)

Этот проект реализует pipeline для извлечения сущностей из PDF-документов с использованием **локальной LLM** через **Ollama**, без подключения к OpenAI (по умолчанию). Поддерживается переключение между API и локальной моделью.

---

## ⚙️ Обновления в этой версии

- ✅ **Локальный LLM через Ollama** (phi3 или mistral, работает на CPU)
- ✅ Переключение между API-моделью и локальной (через `.env`)
- ✅ Готовый `llm_setup.py` для LangChain + Ollama
- ✅ `docker-compose.yml` запускает Ollama сервер
- ✅ Возможность теста на локальных PDF без подключения к интернету

---

## 🚀 Быстрый запуск (локально через Docker Compose)

### 📦 1. Клонируй проект:
```bash
git clone https://github.com/your/repo.git
cd your-repo
```

### 🔑 2. Создай `.env`:
```bash
cp .env.example .env
```

И задай режим LLM:
```
LLM_MODE=ollama        # или openai
OLLAMA_MODEL=phi3      # если используешь локальную
OPENAI_MODEL=gpt-4     # если используешь API
```

### 🐳 3. Запусти Ollama локально:
```bash
docker-compose up -d ollama
```

### 🧠 4. Загрузить модель:
```bash
docker exec -it ollama ollama run phi3
```
> Или замени `phi3` на другую компактную модель: `mistral`, `llama3`, `gemma` и др.

### 🧪 5. Запусти агент:
```bash
python src/pdf_llm_agent_pipeline.py pdfs/contract1.pdf
```

---

## 🧠 Пример `llm_setup.py` с переключением моделей

```python
# src/llm_setup.py
import os

LLM_MODE = os.getenv("LLM_MODE", "ollama")

if LLM_MODE == "openai":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4"))

elif LLM_MODE == "ollama":
    from langchain_community.llms import Ollama
    llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "phi3"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
else:
    raise ValueError(f"Неизвестный LLM_MODE: {LLM_MODE}")
```

---

## 🐳 Пример docker-compose.yml

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    restart: always

volumes:
  ollama:
```

---

## 📂 Пример запуска на PDF

```bash
python src/pdf_llm_agent_pipeline.py pdfs/contract1.pdf
```

**Пример вывода:**
```json
{
  "inn": "7707083893",
  "date": "2023-09-15",
  "has_stamp": true,
  "mentions_guarantee": false
}
```

И автоматически отправляется через `result_sender.py` на внешний REST API, если он настроен.

---

## 🧪 Тест PDF

Папка `pdfs/` предназначена для ПДФ файлов:
```
pdfs/
├── contract1.pdf
├── contract2.pdf
```

запускай `pdf_llm_agent_pipeline.py`

---
---

=============================================================

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

'''
Для рестарта с удалением кэша и образоввщ:
docker-compose down --volumes
docker system prune -af
docker-compose build --no-cache
docker-compose up

'''

## 📥 Эндпоинты FastAPI

### 1. `POST /upload-pdf`
Загрузка PDF-файла:
```bash
curl -X 'POST' \
  'http://localhost:8000/upload-pdf' \
  -H 'accept: application/json' \
  -F 'file=@sample.pdf' \
  -F 'ground_truth={"inn": "1234567890", "has_stamp": true}'
```

### 2. `POST /upload-text`
Загрузка обычного текста:
```bash
curl -X 'POST' \
  'http://localhost:8000/upload-text' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'text=ИНН 1234567890 содержится в этом тексте' \
  -d 'ground_truth={"inn": "1234567890"}'
```

## 📌 Примеры запросов на Python (requests)

### ✅ Отправка PDF-файла с `ground_truth`:
```
import requests

url = "http://localhost:8000/upload-pdf"
file_path = "sample.pdf"

with open(file_path, "rb") as f:
    files = {"file": ("sample.pdf", f, "application/pdf")}
    data = {
        "ground_truth": '{"inn": "1234567890", "has_stamp": true, "mentions_guarantee": false}'
    }
    response = requests.post(url, files=files, data=data)

print("Status:", response.status_code)
print("Response:", response.json())
```

### ✅ Отправка текста с `ground_truth`:
```
import requests

url = "http://localhost:8000/upload-text"

data = {
    "text": "Это пример текста, содержащего ИНН 1234567890 и упоминание гарантий.",
    "ground_truth": '{"inn": "1234567890", "mentions_guarantee": true}'
}

response = requests.post(url, data=data)

print("Status:", response.status_code)
print("Response:", response.json())
```

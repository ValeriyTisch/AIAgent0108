import os
import requests
import logging

logger = logging.getLogger(__name__)

POST_RESULT_URL = os.getenv("POST_RESULT_URL")
POST_RESULT_AUTH_TOKEN = os.getenv("POST_RESULT_AUTH_TOKEN")

def send_result_to_api(result: dict, metadata: dict = None):
    if not POST_RESULT_URL:
        logger.warning("POST_RESULT_URL не задан — результат не отправлен.")
        return

    headers = {"Content-Type": "application/json"}
    if POST_RESULT_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {POST_RESULT_AUTH_TOKEN}"

    payload = result.copy()
    if metadata:
        payload.update(metadata)

    try:
        response = requests.post(POST_RESULT_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ Результат успешно отправлен: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"❌ Ошибка при отправке результата: {e}")

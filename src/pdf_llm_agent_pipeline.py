import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from src.models import ExtractedEntities
from src.retriever_pgvector import retriever
from src.llm_setup import llm
from src.result_sender import send_result_to_api

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def query_llm(chunk: str) -> str:
    context_docs = retriever.get_relevant_documents(chunk)
    context = "\n".join([doc.page_content for doc in context_docs])
    prompt = f"Контекст:\n{chunk}\n\nОпредели наличие следующих признаков:\n{context}"
    return llm.invoke(prompt)

def run_agent_on_text(text: str, ground_truth: dict = None) -> dict:
    chunks = [text]  # при необходимости заменить на чанкинг
    try:
        response = query_llm(chunks[0])
        parsed = ExtractedEntities.parse_raw(response.content if hasattr(response, 'content') else response)
        result = parsed.dict()
        logger.info(f"Распознанные entities: {result}")
    except Exception as e:
        logger.error(f"Ошибка парсинга LLM, итогового JSON не будет: {e}")
        raise

    send_result_to_api(result)

    if ground_truth:
        result['ground_truth'] = ground_truth

    return result


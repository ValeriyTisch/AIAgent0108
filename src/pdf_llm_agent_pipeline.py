from result_sender import send_result_to_api
def run_agent_on_text(text: str, ground_truth: dict = None):
    import uuid
    import logging
    from tenacity import retry, stop_after_attempt, wait_fixed
    from text_extraction import chunk_text
    from retriever_pgvector import PGEnsembleRetriever
    from llm_setup import llm
    from models import ExtractedEntities
    from eval_utils import evaluate_output
    from phoenix_setup import get_arize_logger

    logger = logging.getLogger(__name__)
    retriever = PGEnsembleRetriever()
    chunks = chunk_text(text)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def query_llm(chunk):
        context = retriever.get_context(chunk)
        prompt = f"Контекст:\n{chunk}\n\nОпредели наличие следующих признаков:\n{context}"
        return llm.invoke(prompt)

    response = query_llm(chunks[0])
    try:
        parsed = ExtractedEntities.parse_raw(response.content)
        result = parsed.dict()
        # где `entities` — результат от LLM после pydantic-валидации
        send_result_to_api(result, metadata={"source": "pdf-agent"})
        # return result
    except Exception as e:
        logger.error(f"Ошибка парсинга LLM: {e}")
        raise

    if ground_truth:
        metrics = evaluate_output(result, ground_truth)
        logger.info(f"Метрики: {metrics}")

    get_arize_logger()(prediction_id=str(uuid.uuid4()), prompt=chunks[0], llm_output=result, ground_truth=ground_truth)
    return result

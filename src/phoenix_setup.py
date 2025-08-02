def start_phoenix():
    from phoenix.trace.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument()

def get_arize_logger():
    import phoenix as px
    return px.log

def get_logging_schema():
    import phoenix as px
    return px.Schema(
        prompt_column_name="prompt",
        prediction_id_column_name="prediction_id",
        response_column_name="llm_output",
        actual_column_name="ground_truth"
    )
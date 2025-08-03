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
    raise ValueError(f"Unknown LLM_MODE: {LLM_MODE}")
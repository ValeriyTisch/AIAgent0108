# 📁 src/retriever_pgvector.py

import os
import json
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from sqlalchemy import create_engine, text
from typing import List

class PGEnsembleRetriever:
    def __init__(self, collection_name: str = "entity_descriptions"):
        self.embeddings = OpenAIEmbeddings()
        self.pg_url = os.getenv("PGVECTOR_URL", "postgresql+psycopg2://user:password@localhost:5432/mydb")
        self._init_pgvector_extension()

        self.store = PGVector(
            collection_name=collection_name,
            connection_string=self.pg_url,
            embedding_function=self.embeddings
        )
        self._autoload_descriptions()

    def _init_pgvector_extension(self):
        try:
            engine = create_engine(self.pg_url)
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as e:
            print(f"Ошибка инициализации расширения pgvector: {e}")

    def _autoload_descriptions(self):
        path = "data/entity_descriptions.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")
        with open(path, "r", encoding="utf-8") as f:
            descriptions = json.load(f)
        self.add_entity_descriptions(descriptions)

    def _exists(self, entity_key: str) -> bool:
        results = self.store.similarity_search(entity_key, k=5)
        for doc in results:
            if doc.metadata.get("entity") == entity_key:
                return True
        return False

    def add_entity_descriptions(self, descriptions: dict):
        docs = []
        for key, desc in descriptions.items():
            if not self._exists(key):
                docs.append(Document(page_content=desc, metadata={"entity": key}))
        if docs:
            self.store.add_documents(docs)

    def get_context(self, query: str, top_k: int = 5) -> str:
        results = self.store.similarity_search(query, k=top_k)
        return "\n".join([doc.page_content for doc in results])

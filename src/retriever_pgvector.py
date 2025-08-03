# src/retriever_pgvector.py
import os
import psycopg2
from dotenv import load_dotenv
from langchain.vectorstores.pgvector import PGVector
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# Настройка эмбеддингов
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small")

# Подключение к PostgreSQL
pgvector_connection_string = f"postgresql+psycopg2://{os.getenv('PGVECTOR_USER', 'postgres')}:{os.getenv('PGVECTOR_PASSWORD', 'postgres')}@{os.getenv('PGVECTOR_HOST', 'localhost')}:{os.getenv('PGVECTOR_PORT', 5432)}/{os.getenv('PGVECTOR_DB', 'postgres')}"

# Создание retriever через from_existing_index
collection_name = os.getenv("PGVECTOR_COLLECTION", "entities")
entity_pg = PGVector.from_existing_index(
    embedding=embedding_model,
    collection_name=collection_name,
    connection_string=pgvector_connection_string
)

retriever = entity_pg.as_retriever()

def load_entity_descriptions():
    # Проверка: если есть хотя бы 1 документ, пропускаем
    if entity_pg.similarity_search("ИНН", k=1):
        print("✅ Описания сущностей уже загружены в PGVector")
        return

    entity_descriptions = [
        {"name": "has_stamp", "description": "Признак наличия печати в документе"},
        {"name": "mentions_guarantee", "description": "Упоминается ли гарантия или обязательство"},
        {"name": "contains_penalty", "description": "Указания на штрафные санкции или ответственность"}
    ]

    texts = [f"{e['name']}: {e['description']}" for e in entity_descriptions]
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.create_documents(texts)

    # Проверка на дубли и добавление
    for i, doc in enumerate(docs):
        name = entity_descriptions[i]['name']
        if not entity_pg.similarity_search(name, k=1):
            PGVector.from_documents(
                documents=[doc],
                embedding=embedding_model,
                collection_name=collection_name,
                connection_string=pgvector_connection_string
            )

load_entity_descriptions()

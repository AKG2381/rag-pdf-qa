from functools import lru_cache

from langchain_chroma import Chroma

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_retriever(collection: str = "default"):
    store = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.vectorstore_path,
    )
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retrieval_k, "fetch_k": settings.retrieval_k * 3},
    )

# src/embeddings/factory.py

import os

from src.embeddings.ollama_embedding import OllamaEmbedding
from src.embeddings.bge_embedding import BGEEmbedding
from src.embeddings.openai_embedding import OpenAIEmbedding


def get_embedding_provider():
    provider = os.getenv(
        "EMBED_PROVIDER",
        "ollama",
    ).lower()

    if provider == "ollama":
        return OllamaEmbedding()

    if provider == "bge":
        return BGEEmbedding()

    if provider == "openai":
        return OpenAIEmbedding()

    raise ValueError(
        f"Unsupported embedding provider: {provider}"
    )
# src/embeddings/bge_embedding.py

import os

from sentence_transformers import SentenceTransformer


class BGEEmbedding:
    """
    Local BGE embedding provider.

    This class has the same interface as OllamaEmbedding:

    - embed_query(text)
    - embed_documents(texts)

    It does NOT require:
    - Ollama
    - HTTP server
    - http://localhost:5001
    """

    def __init__(self):
        self.model_name = os.getenv(
            "BGE_EMBED_MODEL",
            "BAAI/bge-base-en-v1.5",
        )

        self.model = SentenceTransformer(
            self.model_name
        )

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()
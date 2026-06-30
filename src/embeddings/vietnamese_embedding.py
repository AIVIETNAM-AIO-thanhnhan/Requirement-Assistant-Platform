# src/embeddings/vietnamese_embedding.py

import os

from sentence_transformers import SentenceTransformer


class VietnameseEmbedding:
    """
    Local Vietnamese embedding provider using AITeamVN/Vietnamese_Embedding.

    Built on BGE-M3 (1024-dim, dot-product similarity).
   
    Usage:
        EMBED_PROVIDER=vietnamese
        VI_EMBED_MODEL=AITeamVN/Vietnamese_Embedding  # optional override
    """

    def __init__(self):
        self.model_name = os.getenv(
            "VI_EMBED_MODEL",
            "AITeamVN/Vietnamese_Embedding",
        )

        self.model = SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

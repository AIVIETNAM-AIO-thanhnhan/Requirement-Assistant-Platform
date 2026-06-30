import os
from typing import List

from langchain_ollama import OllamaEmbeddings


class OllamaEmbedding:
    """
    Ollama embedding provider.

    Expected .env:
        OLLAMA_URL=http://localhost:11434
        OLLAMA_EMBED_MODEL=nomic-embed-text
    """

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

        self.embedding_client = OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
        )

    def embed(self, text: str) -> List[float]:
        return self.embedding_client.embed_query(text)

    def embed_documents(self, texts: List[str], batch_size: int = 200) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            results.extend(self.embedding_client.embed_documents(texts[i:i + batch_size]))
        return results

    def embed_query(self, query: str) -> List[float]:
        return self.embedding_client.embed_query(query)
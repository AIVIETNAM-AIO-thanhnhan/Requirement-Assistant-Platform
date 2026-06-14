import os
from typing import List

from openai import OpenAI


class OpenAIEmbedding:
    """
    OpenAI embedding provider.

    Expected .env:

        OPENAI_API_KEY=xxxxx
        OPENAI_EMBED_MODEL=text-embedding-3-small
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.model = os.getenv(
            "OPENAI_EMBED_MODEL",
            "text-embedding-3-small",
        )

        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Embed multiple documents.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]

    def embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Embed a user question.
        """
        return self.embed(query)
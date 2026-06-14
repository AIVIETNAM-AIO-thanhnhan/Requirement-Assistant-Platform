# src/vectordb/retriever.py

import os

from src.vectordb.chroma_store import ChromaStore


class Retriever:

    def __init__(self):

        self.top_k = int(
            os.getenv(
                "TOP_K",
                "5"
            )
        )

        self.store = ChromaStore()

    def retrieve(
        self,
        query_embedding
    ):

        results = self.store.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )

        return results
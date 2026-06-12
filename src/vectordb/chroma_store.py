# src/vectordb/chroma_store.py

"""
chroma_store.py

Purpose
-------
Wrapper around ChromaDB for storing and querying document chunks.

This class is responsible for:
1. Creating/loading a persistent ChromaDB database
2. Creating/loading a collection
3. Adding or updating embedded document chunks
4. Querying similar chunks
5. Resetting the collection during development
"""

import os

import chromadb
from dotenv import load_dotenv

load_dotenv()


class ChromaStore:
    """
    Local persistent ChromaDB store.

    Expected .env:

        CHROMA_DB_PATH=./data/chroma
        CHROMA_COLLECTION=qa_documents
    """

    def __init__(self):
        self.db_path = os.getenv(
            "CHROMA_DB_PATH",
            "./data/chroma",
        )

        self.collection_name = os.getenv(
            "CHROMA_COLLECTION",
            "qa_documents",
        )

        # PersistentClient stores database files locally.
        # Example:
        # data/chroma/chroma.sqlite3
        self.client = chromadb.PersistentClient(
            path=self.db_path,
        )

        # Create collection if it does not exist.
        # Load collection if it already exists.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ):
        """
        Add new documents to ChromaDB.

        Use this only when IDs are new.
        If the same ID already exists, ChromaDB may raise an error.
        """

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def upsert_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ):
        """
        Add or update documents in ChromaDB.

        This is safer than add_documents() during development
        because you can rerun build_vectordb.py without duplicate ID errors.
        """

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict:
        """
        Query ChromaDB using an embedding vector.
        """

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        """
        Return number of chunks stored in the collection.
        """

        return self.collection.count()

    def delete_collection(self):
        """
        Delete the current collection.

        Useful for resetting data during development.
        """

        self.client.delete_collection(
            name=self.collection_name,
        )
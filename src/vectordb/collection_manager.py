# src/vectordb/collection_manager.py

from src.vectordb.chroma_store import ChromaStore


class CollectionManager:

    def __init__(self):

        self.store = ChromaStore()

    def stats(self):

        return {
            "collection": self.store.collection_name,
            "count": self.store.count()
        }

    def reset(self):

        self.store.delete_collection()
# scripts/build_vectordb.py

"""
build_vectordb.py

Purpose
-------
Build the local ChromaDB vector database from processed document chunks.

Input
-----
data/processed/chunks.json

Usage
-----
python scripts/build_vectordb.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from src.embeddings.factory import get_embedding_provider
from src.vectordb.chroma_store import ChromaStore

load_dotenv()

CHUNKS_FILE = Path("data/processed/chunks.json")


def load_chunks() -> list[dict]:
    """
    Load processed chunks from JSON file.
    """

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {CHUNKS_FILE}. "
            "Please run scripts/process_documents.py first "
            "or create data/processed/chunks.json manually."
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("chunks.json is empty.")

    return chunks


def prepare_chroma_payload(
    chunks: list[dict],
) -> tuple[list[str], list[str], list[dict]]:
    """
    Convert chunks.json data into ChromaDB input format.
    """

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()

        if not text:
            print(f"Skip empty chunk at index {index}")
            continue

        chunk_id = chunk.get("chunk_id", f"chunk_{index}")

        ids.append(chunk_id)
        documents.append(text)

        metadatas.append(
            {
                "source": chunk.get("source", ""),
                "page": chunk.get("page", ""),
                "document_type": chunk.get("document_type", ""),
                "module": chunk.get("module", ""),
            }
        )

    if not documents:
        raise ValueError("No valid chunks found to store in ChromaDB.")

    return ids, documents, metadatas


def main():
    """
    Main flow:
    1. Load chunks
    2. Create embeddings
    3. Save vectors to ChromaDB
    """

    chunks = load_chunks()

    ids, documents, metadatas = prepare_chroma_payload(chunks)

    embedding_provider = get_embedding_provider()
    store = ChromaStore()

    print(f"Creating embeddings for {len(documents)} chunks...")
    embeddings = embedding_provider.embed_documents(documents)

    print("Adding documents to ChromaDB...")

    store.upsert_documents(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {len(documents)} chunks.")
    print(f"Collection count: {store.count()}")


if __name__ == "__main__":
    main()
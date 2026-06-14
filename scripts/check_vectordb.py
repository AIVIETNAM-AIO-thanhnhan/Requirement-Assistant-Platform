# scripts/check_vectordb.py

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from src.vectordb.chroma_store import ChromaStore

load_dotenv()


def main():
    store = ChromaStore()

    print("Collection Information")
    print("----------------------")

    print(
        f"Collection count: {store.count()}"
    )

    results = store.collection.peek(5)

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    print("\nSample Documents")
    print("----------------------")

    for index, document in enumerate(documents):
        print(
            f"\nDocument {index + 1}"
        )

        print(document[:300])

        if index < len(metadatas):
            print(
                f"Metadata: "
                f"{metadatas[index]}"
            )


if __name__ == "__main__":
    main()
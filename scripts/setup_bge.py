# scripts/setup_bge.py

"""
setup_bge.py

Purpose
-------
Setup and verify local BGE embedding environment.

Important
---------
This script does NOT start an HTTP server.

BGE is loaded directly in Python using sentence-transformers:

    SentenceTransformer("BAAI/bge-base-en-v1.5")

So normally you do NOT need:

    http://localhost:5001

Use BGE_SERVER_URL only if you later build a separate FastAPI
embedding server.

This script:
1. Checks required Python packages
2. Downloads/loads BGE embedding model
3. Runs a small embedding test
4. Prints embedding dimension
5. Explains how to configure .env

Usage
-----
python scripts/setup_bge.py

Recommended .env
----------------
EMBED_PROVIDER=bge
BGE_EMBED_MODEL=BAAI/bge-base-en-v1.5

Optional only for future BGE API server:
BGE_SERVER_URL=http://localhost:5001
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

BGE_EMBED_MODEL = os.getenv(
    "BGE_EMBED_MODEL",
    "BAAI/bge-base-en-v1.5",
)


def check_package(package_name: str) -> bool:
    """
    Check whether a Python package is installed.
    """

    try:
        __import__(package_name)
        print(f"Package installed: {package_name}")
        return True

    except ImportError:
        print(f"Missing package: {package_name}")
        return False


def verify_required_packages() -> bool:
    """
    Verify packages required for local BGE embeddings.
    """

    required_packages = [
        "sentence_transformers",
        "torch",
    ]

    missing_packages = []

    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)

    if missing_packages:
        print("\nMissing required packages.")
        print("Please install them with:")
        print("pip install sentence-transformers torch")
        return False

    return True


def setup_bge_model():
    """
    Download/load BGE model using sentence-transformers.

    The model will be cached locally after the first download.
    """

    from sentence_transformers import SentenceTransformer

    print("\nLoading BGE model locally...")
    print(f"Model: {BGE_EMBED_MODEL}")

    model = SentenceTransformer(BGE_EMBED_MODEL)

    print("BGE model loaded successfully.")

    return model


def run_embedding_test(model):
    """
    Run a small embedding test.
    """

    test_texts = [
        "The system shall lock the user account after 5 failed login attempts.",
        "Users shall be able to reset their password using a verified email address.",
    ]

    print("\nRunning embedding test...")

    embeddings = model.encode(
        test_texts,
        normalize_embeddings=True,
    )

    print(f"Number of test texts: {len(test_texts)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print("Embedding test completed successfully.")


def print_configuration_note():
    """
    Print configuration guidance for team members.
    """

    print("\nBGE setup completed successfully.")
    print("\nTo use BGE embeddings, set this in .env:")
    print("EMBED_PROVIDER=bge")
    print(f"BGE_EMBED_MODEL={BGE_EMBED_MODEL}")

    print("\nImportant note:")
    print("BGE is running locally inside Python.")
    print("You do NOT need to open http://localhost:5001.")
    print("You do NOT need BGE_SERVER_URL unless you build a BGE API server later.")

    print("\nNext commands:")
    print("python scripts/benchmark_embeddings.py")
    print("python scripts/build_vectordb.py")


def main():
    print("Setting up BGE embedding environment...")
    print(f"BGE model: {BGE_EMBED_MODEL}")

    if not verify_required_packages():
        return

    model = setup_bge_model()

    run_embedding_test(model)

    print_configuration_note()


if __name__ == "__main__":
    main()
# scripts/setup_project.py

"""
setup_project.py

Purpose
-------
Prepare the local Requirement Assistant Platform environment.

Modes
-----

Developer Mode:
    python scripts/setup_project.py

    - Create project folders
    - Setup selected embedding provider
    - Build ChromaDB if processed chunks exist

Demo Mode:
    python scripts/setup_project.py --demo

    - Everything in Developer Mode
    - Create benchmark datasets
    - Prepare sample data for evaluation

Embedding Provider Examples
---------------------------

Ollama:
    python scripts/setup_project.py --embed-provider ollama

BGE:
    python scripts/setup_project.py --embed-provider bge

OpenAI:
    python scripts/setup_project.py --embed-provider openai

Demo with BGE:
    python scripts/setup_project.py --demo --embed-provider bge
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CHUNKS_FILE = Path("data/processed/chunks.json")

BENCHMARK_REQUIRED_FILES = [
    Path("data/benchmark/requirement_embedding_pairs.jsonl"),
    Path("data/benchmark/requirement_chunks.jsonl"),
    Path("data/benchmark/retrieval_eval.jsonl"),
    Path("data/benchmark/llm_artifact_eval.jsonl"),
]


def create_project_folders():
    """
    Create required project folders for data, ChromaDB, and outputs.
    """

    folders = [
        "data/raw",
        "data/processed",
        "data/benchmark",
        "data/chroma",
        "outputs",
        "outputs/embeddings/reports",
        "outputs/embeddings/images",
        "outputs/llms/reports",
        "outputs/llms/images",
        "outputs/experiments",
    ]

    for folder in folders:
        Path(folder).mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Project folders created.")


def run_script(script_path: str):
    """
    Run another Python script using the current Python interpreter.
    """

    print(f"\nRunning: {script_path}")

    subprocess.run(
        [sys.executable, script_path],
        check=True,
    )


def check_chunks_file_exists() -> bool:
    """
    Check whether processed chunks are available.
    """

    return CHUNKS_FILE.exists()


def benchmark_files_exist() -> bool:
    """
    Check whether all required benchmark files already exist.
    """

    return all(
        file_path.exists()
        for file_path in BENCHMARK_REQUIRED_FILES
    )


def setup_benchmark_data_if_needed():
    """
    Create benchmark datasets if they do not already exist.
    """

    if benchmark_files_exist():
        print("\nBenchmark data already exists.")
        return

    print("\nBenchmark data not found. Creating sample benchmark data...")
    run_script("scripts/setup_benchmark_data.py")


def setup_embedding_provider(provider: str):
    """
    Setup selected embedding provider.
    """

    provider = provider.lower()

    print(f"\nEmbedding provider selected: {provider}")

    if provider == "ollama":
        os.environ["EMBED_PROVIDER"] = "ollama"
        run_script("scripts/setup_ollama.py")
        return

    if provider == "bge":
        os.environ["EMBED_PROVIDER"] = "bge"
        run_script("scripts/setup_bge.py")
        return

    if provider == "openai":
        os.environ["EMBED_PROVIDER"] = "openai"

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in .env. "
                "Please set OPENAI_API_KEY before using --embed-provider openai."
            )

        print("OpenAI API key found.")
        print("OpenAI embedding setup completed.")
        return

    raise ValueError(
        f"Unsupported embedding provider: {provider}. "
        "Use: ollama, bge, or openai."
    )


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Setup Requirement Assistant Platform environment."
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create sample benchmark data for demo and benchmark runs.",
    )

    parser.add_argument(
        "--embed-provider",
        choices=[
            "ollama",
            "bge",
            "openai",
        ],
        default=os.getenv("EMBED_PROVIDER", "ollama"),
        help="Embedding provider to setup: ollama, bge, or openai.",
    )

    return parser.parse_args()


def main():
    """
    Full setup flow:

        Create folders
            ↓
        Setup selected embedding provider
            ↓
        Optional: create benchmark data
            ↓
        Optional: build ChromaDB if chunks.json exists
            ↓
        Ready
    """

    args = parse_args()

    print("Starting full project setup...")

    create_project_folders()

    setup_embedding_provider(args.embed_provider)

    if args.demo:
        setup_benchmark_data_if_needed()

    if check_chunks_file_exists():
        run_script("scripts/build_vectordb.py")
    else:
        print("\nSkip ChromaDB build.")
        print(f"Cannot find: {CHUNKS_FILE}")
        print(
            "Run scripts/process_documents.py first "
            "or create chunks.json manually."
        )

    print("\nProject setup completed successfully.")
    print(f"Embedding provider: {args.embed_provider}")


if __name__ == "__main__":
    main()
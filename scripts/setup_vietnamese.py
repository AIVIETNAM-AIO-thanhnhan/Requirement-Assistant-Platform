# scripts/setup_vietnamese.py

"""
Setup and verify the Vietnamese embedding environment.

This script:
1. Checks required Python packages (sentence-transformers, torch)
2. Downloads/loads AITeamVN/Vietnamese_Embedding from HuggingFace
3. Runs a small Vietnamese embedding test
4. Prints embedding dimension and .env configuration

The model is cached locally after the first download (~1.4 GB).
No HTTP server is needed — it runs directly inside Python.

Usage
-----
python scripts/setup_vietnamese.py

Recommended .env
----------------
EMBED_PROVIDER=vietnamese
VI_EMBED_MODEL=AITeamVN/Vietnamese_Embedding
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

VI_EMBED_MODEL = os.getenv("VI_EMBED_MODEL", "AITeamVN/Vietnamese_Embedding")


def check_package(package_name: str) -> bool:
    try:
        __import__(package_name)
        print(f"  OK  {package_name}")
        return True
    except ImportError:
        print(f"  MISSING  {package_name}")
        return False


def verify_required_packages() -> bool:
    print("Checking required packages...")
    required = ["sentence_transformers", "torch"]
    missing = [p for p in required if not check_package(p)]
    if missing:
        print("\nInstall missing packages with:")
        print("  pip install sentence-transformers torch")
        return False
    return True


def setup_model():
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading model: {VI_EMBED_MODEL}")
    print("(First run will download ~1.4 GB from HuggingFace — this may take a while.)")
    model = SentenceTransformer(VI_EMBED_MODEL)
    print("Model loaded successfully.")
    return model


def run_embedding_test(model):
    test_texts = [
        "Người lao động được quyền đơn phương chấm dứt hợp đồng lao động.",
        "Tiền lương làm thêm giờ được tính theo công thức quy định tại Điều 55.",
    ]

    print("\nRunning Vietnamese embedding test...")
    embeddings = model.encode(test_texts, normalize_embeddings=True)
    print(f"  Test texts   : {len(test_texts)}")
    print(f"  Embedding dim: {len(embeddings[0])}")
    print("Embedding test passed.")


def print_configuration_note():
    print("\nVietnamese embedding setup complete.")
    print("\nAdd to .env:")
    print("  EMBED_PROVIDER=vietnamese")
    print(f"  VI_EMBED_MODEL={VI_EMBED_MODEL}")
    print("\nThis is the baseline model (R@1=0.82, R@3=0.95, MRR=0.88 on 38 Vietnamese queries).")
    print("\nNext steps:")
    print("  python scripts/benchmark_embeddings.py")
    print("  streamlit run app.py")


def main():
    print("Setting up Vietnamese embedding environment")
    print("-------------------------------------------")

    if not verify_required_packages():
        return

    model = setup_model()
    run_embedding_test(model)
    print_configuration_note()


if __name__ == "__main__":
    main()

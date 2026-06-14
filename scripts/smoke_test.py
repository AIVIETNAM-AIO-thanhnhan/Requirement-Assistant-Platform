from dotenv import load_dotenv

from src.embeddings.factory import get_embedding_provider
from src.llm.factory import get_llm_provider
from src.vectordb.chroma_store import ChromaStore

load_dotenv()


def test_embedding():
    embedding_provider = get_embedding_provider()
    vector = embedding_provider.embed_query("What is regression testing?")

    assert isinstance(vector, list)
    assert len(vector) > 0

    print("Embedding provider OK")


def test_chroma():
    store = ChromaStore()
    count = store.count()

    assert isinstance(count, int)

    print("ChromaDB OK")


def test_llm():
    llm = get_llm_provider()
    answer = llm.chat("Say hello in one short sentence.")

    assert isinstance(answer, str)
    assert len(answer) > 0

    print("LLM provider OK")


def main():
    test_embedding()
    test_chroma()
    test_llm()

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
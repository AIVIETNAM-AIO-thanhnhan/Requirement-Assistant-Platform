"""
Streamlit UI for the Requirement Assistant Platform.

Features:
- Upload requirement documents (PDF, DOCX, TXT, XLSX)
- Parse and chunk uploaded content
- Embed and store chunks in ChromaDB
- Ask questions and retrieve relevant sections
- Generate answers with source references
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

SUPPORTED_TYPES = ["pdf", "docx", "txt", "xlsx"]

st.set_page_config(
    page_title="Requirement Assistant",
    page_icon="📋",
    layout="wide",
)


if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "history" not in st.session_state:
    st.session_state.history = []


def _get_chroma_store():
    from src.vectordb.chroma_store import ChromaStore

    return ChromaStore()


def _get_embedding_provider():
    from src.embeddings.factory import get_embedding_provider

    return get_embedding_provider()


def _get_llm_provider():
    from src.llm.factory import get_llm_provider

    return get_llm_provider()


def _get_doc_count() -> int:
    try:
        return _get_chroma_store().count()
    except Exception:
        return 0


def process_uploaded_file(uploaded_file) -> int:
    from src.processing.cleaner import clean_text
    from src.processing.chunker import split_text
    from src.processing.parsers import parse

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {f".{ext}" for ext in SUPPORTED_TYPES}:
        raise ValueError("Unsupported file format.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        raw_text = parse(Path(temp_path))
        if not raw_text or not raw_text.strip():
            raise ValueError("No readable text was found in the uploaded file.")

        cleaned_text = clean_text(raw_text)
        chunks = split_text(cleaned_text, size=900, overlap=120)
        if not chunks:
            raise ValueError("The document did not produce any searchable chunks.")

        embedding_provider = _get_embedding_provider()
        store = _get_chroma_store()

        ids = [f"{Path(uploaded_file.name).stem}-{idx}-{uuid.uuid4().hex[:8]}" for idx in range(len(chunks))]
        documents = chunks
        embeddings = embedding_provider.embed_documents(chunks)
        metadatas = [
            {
                "source": uploaded_file.name,
                "page": "",
                "document_type": suffix.lstrip("."),
                "module": "",
                "chunk_index": idx,
            }
            for idx in range(len(chunks))
        ]

        store.upsert_documents(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def answer_question(question: str):
    embedder = _get_embedding_provider()
    llm = _get_llm_provider()
    store = _get_chroma_store()

    query_embedding = embedder.embed_query(question)
    top_k = int(os.getenv("TOP_K", "5"))

    results = store.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    if not documents:
        return {
            "answer": "I could not find relevant information in the uploaded documents.",
            "sources": [],
            "chunks_used": [],
        }

    context_parts = []
    chunk_details = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        source_name = meta.get("source", "Unknown source") if isinstance(meta, dict) else "Unknown source"
        similarity = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0
        context_parts.append(f"Source: {source_name}\n{doc}")
        chunk_details.append(
            {
                "source": source_name,
                "text": doc,
                "similarity": round(similarity, 3),
                "page": meta.get("page", "Unknown") if isinstance(meta, dict) else "Unknown",
            }
        )

    context = "\n\n".join(context_parts)
    answer = llm.generate_with_context(context, question)

    unique_sources = []
    seen = set()
    for chunk in chunk_details:
        if chunk["source"] not in seen:
            seen.add(chunk["source"])
            unique_sources.append(chunk["source"])

    return {
        "answer": answer,
        "sources": unique_sources,
        "chunks_used": chunk_details,
    }


with st.sidebar:
    st.header("Knowledge Base")
    st.caption("Upload requirements and ask questions about them.")

    if st.session_state.uploaded_docs:
        for doc in st.session_state.uploaded_docs:
            st.write(f"📄 {doc['name']} ({doc['chunk_count']} chunks)")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    if st.button("Clear knowledge base", use_container_width=True):
        try:
            _get_chroma_store().delete_collection()
        except Exception:
            pass
        st.session_state.uploaded_docs = []
        st.session_state.history = []
        st.success("Knowledge base cleared.")
        st.rerun()

    st.divider()
    st.caption(f"Vector count: {_get_doc_count()}")
    st.caption(f"Embedding provider: {os.getenv('EMBED_PROVIDER', 'ollama')}")
    st.caption(f"LLM provider: {os.getenv('LLM_PROVIDER', 'ollama')}")


st.title("Requirement Assistant")
st.markdown(
    "Upload requirement documents, index them into a local vector store, and ask questions grounded in the content you provide."
)

st.markdown("### Supported files")
st.write("PDF, DOCX, TXT, and XLSX files are supported.")

st.divider()

uploaded_files = st.file_uploader(
    "Upload one or more requirement documents",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    help="The app will parse, clean, chunk, and index each file.",
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        already_uploaded = any(doc["name"] == uploaded_file.name for doc in st.session_state.uploaded_docs)
        if already_uploaded:
            st.info(f"{uploaded_file.name} is already indexed.")
            continue

        try:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                chunk_count = process_uploaded_file(uploaded_file)
            st.session_state.uploaded_docs.append(
                {"name": uploaded_file.name, "chunk_count": chunk_count}
            )
            st.success(f"Indexed {uploaded_file.name} into the knowledge base.")
        except Exception as exc:
            st.error(f"Could not index {uploaded_file.name}: {exc}")

st.divider()

st.subheader("Ask a question")
question = st.text_input(
    "Question",
    placeholder="Example: What are the login validation requirements?",
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.uploaded_docs:
        st.warning("Please upload and index at least one document before asking questions.")
    else:
        with st.spinner("Searching the knowledge base..."):
            try:
                result = answer_question(question)
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
                result = None

        if result:
            st.session_state.history.append({"q": question, "a": result["answer"]})
            st.markdown("### Answer")
            st.markdown(result["answer"])

            if result["sources"]:
                st.markdown("### Sources")
                for source in result["sources"]:
                    st.write(f"• {source}")

            if result["chunks_used"]:
                with st.expander("View retrieved sections"):
                    for chunk in result["chunks_used"]:
                        st.markdown(
                            f"**{chunk['source']}** (page {chunk.get('page', 'Unknown')}, similarity {chunk.get('similarity', '—')})"
                        )
                        st.write(chunk["text"])
                        st.divider()

st.divider()

if st.session_state.history:
    st.subheader("Conversation history")
    for entry in reversed(st.session_state.history[-8:]):
        with st.container():
            st.markdown(f"**You:** {entry['q']}")
            st.markdown(entry['a'])
            st.divider()

"""
Streamlit UI for the Requirement Assistant Platform.

Features:
- Upload requirement documents (PDF, DOCX, TXT, XLSX)
- Parse and chunk uploaded content
- Embed and store chunks in ChromaDB
- Ask questions and retrieve relevant sections
- Generate answers with source references
"""

import glob
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(override=True)

SUPPORTED_TYPES = ["pdf", "docx", "txt", "xlsx"]

EMBED_PROVIDER_LABELS = {
    "vietnamese": "Vietnamese Embedding — AITeamVN/Vietnamese_Embedding (Baseline)",
    "bge":        "BGE English — BAAI/bge-base-en-v1.5",
    "ollama":     "Ollama Embedding — nomic-embed-text",
}

LLM_PROVIDER_LABELS = {
    "ollama":  "Ollama (local)",
    "openai":  "OpenAI",
}

st.set_page_config(
    page_title="Requirement Assistant",
    page_icon="📋",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "embed_provider" not in st.session_state:
    st.session_state.embed_provider = os.getenv("EMBED_PROVIDER", "vietnamese")
if "ollama_embed_model" not in st.session_state:
    st.session_state.ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = os.getenv("LLM_PROVIDER", "ollama")
if "ollama_llm_model" not in st.session_state:
    st.session_state.ollama_llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
# Tracks the last-active embed config so on_change callbacks know which collection to drop.
if "_prev_embed_provider" not in st.session_state:
    st.session_state._prev_embed_provider = st.session_state.embed_provider
if "_prev_ollama_embed_model" not in st.session_state:
    st.session_state._prev_ollama_embed_model = st.session_state.ollama_embed_model


# ── Settings helpers ──────────────────────────────────────────────────────────

def _apply_settings():
    """Push current UI selections into os.environ before any factory call."""
    ep = st.session_state.embed_provider
    os.environ["EMBED_PROVIDER"] = ep
    # Each embedder gets its own ChromaDB collection to avoid dimension conflicts.
    os.environ["CHROMA_COLLECTION"] = f"qa_documents_{ep}"
    os.environ["OLLAMA_EMBED_MODEL"] = st.session_state.ollama_embed_model
    os.environ["LLM_PROVIDER"] = st.session_state.llm_provider
    os.environ["OLLAMA_LLM_MODEL"] = st.session_state.ollama_llm_model


def _delete_collection(collection_name: str):
    """Delete a ChromaDB collection by name, silently ignore if it doesn't exist."""
    try:
        import chromadb
        db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma")
        client = chromadb.PersistentClient(path=db_path)
        client.delete_collection(collection_name)
    except Exception:
        pass


def _on_embed_provider_change():
    """Delete the old collection and reset session when embedding provider changes."""
    old_collection = f"qa_documents_{st.session_state._prev_embed_provider}"
    _delete_collection(old_collection)
    st.session_state._prev_embed_provider = st.session_state.embed_provider
    st.session_state._prev_ollama_embed_model = st.session_state.ollama_embed_model
    st.session_state.history = []


def _on_ollama_embed_model_change():
    """Delete the ollama collection and reset session when the specific model changes."""
    _delete_collection("qa_documents_ollama")
    st.session_state._prev_ollama_embed_model = st.session_state.ollama_embed_model
    st.session_state.history = []


@st.cache_data(ttl=15)
def _get_ollama_models() -> list[str]:
    try:
        import requests
        r = requests.get(
            f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/tags",
            timeout=3,
        )
        if r.ok:
            return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        pass
    return []


def _resolve_model_index(models: list[str], wanted: str, prefer_embed: bool = False) -> int:
    """Best-match index for `wanted`, tolerant of Ollama ':tag' suffixes.

    Ollama lists models tagged (e.g. ``nomic-embed-text:latest``) while .env
    stores the bare name (``nomic-embed-text``). An exact-only match silently
    falls back to index 0 — which can be a chat model that cannot embed and
    raises "does not support embeddings". Match on the base name first, and for
    the embedding selector prefer a model whose name looks like an embedder.
    """
    if wanted in models:
        return models.index(wanted)
    base = wanted.split(":")[0]
    for i, m in enumerate(models):
        if m.split(":")[0] == base:
            return i
    if prefer_embed:
        for i, m in enumerate(models):
            if "embed" in m.lower():
                return i
    return 0


# Apply settings at the start of every rerun.
_apply_settings()


# ── Provider helpers ──────────────────────────────────────────────────────────

def load_sample_questions():
    import json
    path = ROOT_DIR / "eval" / "gold_match.jsonl"
    questions = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = row.get("query") or row.get("question")
                if q:
                    questions.append(q)
    except Exception:
        pass
    return questions


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


def _get_indexed_docs() -> list[dict]:
    """Read the indexed documents straight from the vector store (source of truth)."""
    try:
        return _get_chroma_store().list_documents()
    except Exception:
        return []


@st.cache_data(ttl=30)
def _load_benchmark_metrics(kind: str, model_name: str) -> dict:
    """Parse the Executive Summary metrics from a model's benchmark report.

    ``kind`` is "embeddings" or "llms". Report folders are named after the
    sanitized model name (``/`` and ``.`` → ``_``), e.g. ``BAAI/bge-base-en-v1.5``
    → ``BAAI_bge-base-en-v1_5`` and ``llama3.2`` → ``llama3_2``.
    Returns an ordered {metric: value} dict, or {} when no report exists.
    """
    if not model_name:
        return {}

    safe = re.sub(r"[/.]", "_", model_name)
    report_name = (
        "embedding_summary_report.md"
        if kind == "embeddings"
        else "answer_quality_summary_report.md"
    )
    pattern = str(ROOT_DIR / "outputs" / kind / "*" / safe / "reports" / report_name)
    matches = glob.glob(pattern)
    if not matches:
        return {}

    metrics: dict[str, str] = {}
    in_summary = False
    try:
        with open(matches[0], encoding="utf-8") as f:
            for line in f:
                if line.startswith("## 2. Executive Summary"):
                    in_summary = True
                    continue
                if in_summary and line.startswith("## "):
                    break
                if in_summary:
                    m = re.match(r"\s*-\s*(.+?)\s*=\s*\*\*(.+?)\*\*", line)
                    if m:
                        metrics[m.group(1).strip()] = m.group(2).strip()
    except Exception:
        return {}
    return metrics


@st.cache_data(ttl=30)
def _load_all_benchmarks(kind: str) -> list[dict]:
    """Collect every model's benchmark metrics for side-by-side comparison.

    Every model is benchmarked against the same fixed corpus + gold set in
    data/benchmark/, so these rows are directly comparable. Returns a list of
    {"Model": name, <metric>: value} dicts, one per benchmarked model.
    """
    report_name = (
        "embedding_summary_report.md"
        if kind == "embeddings"
        else "answer_quality_summary_report.md"
    )
    model_label = "Embedding Model" if kind == "embeddings" else "LLM Model"
    pattern = str(ROOT_DIR / "outputs" / kind / "*" / "*" / "reports" / report_name)

    rows = []
    for report_path in sorted(glob.glob(pattern)):
        path = Path(report_path)
        model = path.parts[-3]  # outputs/<kind>/<provider>/<model_safe>/reports/x.md
        metrics: dict[str, str] = {}
        in_summary = False
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = re.match(rf"\|\s*{re.escape(model_label)}\s*\|\s*(.+?)\s*\|", line)
                if row:
                    model = row.group(1).strip()
                if line.startswith("## 2. Executive Summary"):
                    in_summary = True
                    continue
                if in_summary and line.startswith("## "):
                    in_summary = False
                if in_summary:
                    m = re.match(r"\s*-\s*(.+?)\s*=\s*\*\*(.+?)\*\*", line)
                    if m:
                        metrics[m.group(1).strip()] = m.group(2).strip()
        except Exception:
            continue
        rows.append({"Model": model, **metrics})
    return rows


@st.cache_data(ttl=30)
def _load_rag_eval_comparison() -> list[dict]:
    """Collect end-to-end RAG retrieval scores (scripts/rag_eval.py) per embedder.

    Unlike the component benchmark, these queries run against the REAL ChromaDB
    corpus using eval/gold_match.jsonl, so they reveal how each embedder performs
    on the actual (Vietnamese) documents. Reads outputs/comparison/rag_eval_*.json.
    """
    import json

    rows = []
    pattern = str(ROOT_DIR / "outputs" / "comparison" / "rag_eval_*.json")
    for path in sorted(glob.glob(pattern)):
        provider = Path(path).stem.replace("rag_eval_", "")
        try:
            summary = json.loads(Path(path).read_text(encoding="utf-8")).get("summary", {})
        except Exception:
            continue

        def _pct(key):
            v = summary.get(key)
            return f"{v:.2%}" if isinstance(v, (int, float)) else "—"

        rows.append({
            "Provider": provider,
            "Recall@1": _pct("recall@1"),
            "Recall@3": _pct("recall@3"),
            "Recall@5": _pct("recall@5"),
            "MRR": f"{summary['mrr']:.4f}" if isinstance(summary.get("mrr"), (int, float)) else "—",
            "Queries": summary.get("n_queries", "—"),
        })
    return rows


# ── Core logic ────────────────────────────────────────────────────────────────

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
        chunks = split_text(cleaned_text, size=1000, overlap=150)
        if not chunks:
            raise ValueError("The document did not produce any searchable chunks.")

        embedding_provider = _get_embedding_provider()
        store = _get_chroma_store()

        ids = [
            f"{Path(uploaded_file.name).stem}-{idx}-{uuid.uuid4().hex[:8]}"
            for idx in range(len(chunks))
        ]
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
            documents=chunks,
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

# Read the real Knowledge Base contents once per rerun so the sidebar, the
# upload de-dup check, and the Q&A guard all share a single source of truth.
indexed_docs = _get_indexed_docs()

with st.sidebar:
    # ── Model Settings ────────────────────────────────────────────────────────
    with st.expander("⚙️ Model Settings", expanded=False):
        st.markdown("**Embedding**")
        st.selectbox(
            "Provider",
            options=list(EMBED_PROVIDER_LABELS.keys()),
            format_func=lambda x: EMBED_PROVIDER_LABELS[x],
            key="embed_provider",
            on_change=_on_embed_provider_change,
            help="Changing the embedding model clears indexed documents (each model uses its own vector store).",
        )
        if st.session_state.embed_provider == "ollama":
            all_ollama_models = _get_ollama_models()
            # Only embedding-capable models — chat models (llama3.2, ...) cannot
            # embed and raise "does not support embeddings" if selected here.
            ollama_embed_models = [m for m in all_ollama_models if "embed" in m.lower()]
            if ollama_embed_models:
                current_em = st.session_state.ollama_embed_model
                idx = _resolve_model_index(ollama_embed_models, current_em, prefer_embed=True)
                st.selectbox(
                    "Ollama embed model",
                    options=ollama_embed_models,
                    index=idx,
                    key="ollama_embed_model",
                    on_change=_on_ollama_embed_model_change,
                )
            elif all_ollama_models:
                st.text_input("Ollama embed model", key="ollama_embed_model")
                st.caption("⚠️ No embedding model found. Pull one, e.g. `ollama pull nomic-embed-text`.")
            else:
                st.text_input("Ollama embed model", key="ollama_embed_model")
                st.caption("⚠️ Ollama server not reachable — enter model name manually.")

        st.divider()

        st.markdown("**LLM**")
        st.selectbox(
            "Provider",
            options=list(LLM_PROVIDER_LABELS.keys()),
            format_func=lambda x: LLM_PROVIDER_LABELS[x],
            key="llm_provider",
        )
        if st.session_state.llm_provider == "ollama":
            all_ollama_models = _get_ollama_models()
            # Exclude embedding-only models (nomic-embed-text, ...) — they cannot
            # generate chat completions and don't belong in the LLM list.
            ollama_llm_models = [m for m in all_ollama_models if "embed" not in m.lower()]
            if ollama_llm_models:
                current_lm = st.session_state.ollama_llm_model
                idx = _resolve_model_index(ollama_llm_models, current_lm)
                st.selectbox(
                    "Ollama LLM model",
                    options=ollama_llm_models,
                    index=idx,
                    key="ollama_llm_model",
                )
            elif all_ollama_models:
                st.text_input("Ollama LLM model", key="ollama_llm_model")
                st.caption("⚠️ No chat model found. Pull one, e.g. `ollama pull llama3.2`.")
            else:
                st.text_input("Ollama LLM model", key="ollama_llm_model")
                st.caption("⚠️ Ollama server not reachable — enter model name manually.")
        elif st.session_state.llm_provider == "openai":
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if not openai_key:
                st.warning("OPENAI_API_KEY is not set in .env")

        # Reflect current active settings
        st.divider()
        st.caption(f"Active embed: `{os.getenv('EMBED_PROVIDER')}`")
        st.caption(f"Active collection: `{os.getenv('CHROMA_COLLECTION')}`")
        st.caption(f"Active LLM: `{os.getenv('LLM_PROVIDER')}` / `{os.getenv('OLLAMA_LLM_MODEL') if os.getenv('LLM_PROVIDER') == 'ollama' else os.getenv('OPENAI_LLM_MODEL')}`")

    st.divider()

    # ── Knowledge Base ────────────────────────────────────────────────────────
    st.header("Knowledge Base")
    st.caption("Upload requirements and ask questions about them.")

    if indexed_docs:
        for doc in indexed_docs:
            st.write(f"📄 {doc['name']} ({doc['chunk_count']} chunks)")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    if st.button("Reset conversation", width="stretch"):
        st.session_state.history = []
        st.rerun()

    if st.button("Clear knowledge base", width="stretch"):
        try:
            _get_chroma_store().delete_collection()
        except Exception:
            pass
        st.session_state.history = []
        st.success("Knowledge base cleared.")
        st.rerun()

    st.divider()
    st.caption(f"Vector count: {_get_doc_count()}")

    _embed_provider = os.getenv("EMBED_PROVIDER", "")
    _embed_model = {
        "vietnamese": "AITeamVN/Vietnamese_Embedding",
        "bge": "BAAI/bge-base-en-v1.5",
        "ollama": os.getenv("OLLAMA_EMBED_MODEL", ""),
    }.get(_embed_provider, _embed_provider)

    _llm_provider = os.getenv("LLM_PROVIDER", "")
    _llm_model = (
        os.getenv("OLLAMA_LLM_MODEL", "")
        if _llm_provider == "ollama"
        else os.getenv("OPENAI_LLM_MODEL", "")
    )

    st.caption(f"Embedding: `{_embed_model}`")
    st.caption(f"LLM: `{_llm_provider}` / `{_llm_model}`")

    # ── Benchmark metrics for the active models ──────────────────────────────
    _embed_metrics = _load_benchmark_metrics("embeddings", _embed_model)
    _llm_metrics = _load_benchmark_metrics("llms", _llm_model)

    with st.expander("📊 Benchmark metrics"):
        st.markdown(f"**Embedding** · `{_embed_model}`")
        if _embed_metrics:
            for name, value in _embed_metrics.items():
                st.write(f"- {name}: **{value}**")
        else:
            st.caption("No benchmark report for this embedding model.")

        st.divider()

        st.markdown(f"**LLM** · `{_llm_model}`")
        if _llm_metrics:
            for name, value in _llm_metrics.items():
                st.write(f"- {name}: **{value}**")
        else:
            st.caption("No benchmark report for this LLM model.")


# ── Main content ──────────────────────────────────────────────────────────────

st.title("Requirement Assistant")
st.markdown(
    "Upload requirement documents, index them into a local vector store, "
    "and ask questions grounded in the content you provide."
)

st.markdown("### Supported files")
st.write("PDF, DOCX, TXT, and XLSX files are supported.")

st.divider()

# ── Model quality comparison ──────────────────────────────────────────────────
# Every model below was benchmarked against the same corpus + gold set in
# data/benchmark/, so the numbers are directly comparable.
_embed_compare = _load_all_benchmarks("embeddings")
_llm_compare = _load_all_benchmarks("llms")
_rag_compare = _load_rag_eval_comparison()

with st.expander("📊 Model quality comparison", expanded=False):
    st.markdown("**Embedding — component benchmark** (generic `data/benchmark/` set)")
    st.caption(
        "Each embedder scored in isolation on the same fixed sample corpus. "
        "Run `python scripts/benchmark_embeddings.py` to add a model."
    )
    if _embed_compare:
        st.dataframe(_embed_compare, width="stretch", hide_index=True)
    else:
        st.caption("No embedding benchmarks yet.")

    st.divider()

    st.markdown("**Embedding — end-to-end retrieval** (real corpus, `eval/gold_match.jsonl`)")
    st.caption(
        "Queries run against the actual indexed ChromaDB documents — reveals "
        "real-world quality on your Vietnamese corpus. Run "
        "`python scripts/rag_eval.py score --gold eval/gold_match.jsonl` per provider."
    )
    if _rag_compare:
        st.dataframe(_rag_compare, width="stretch", hide_index=True)
    else:
        st.caption("No end-to-end RAG evaluations yet.")

    st.divider()

    st.markdown("**LLM models** — answer quality")
    if _llm_compare:
        st.dataframe(_llm_compare, width="stretch", hide_index=True)
    else:
        st.caption("No LLM benchmarks yet.")

st.divider()

uploaded_files = st.file_uploader(
    "Upload one or more requirement documents",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    help="The app will parse, clean, chunk, and index each file.",
)

if uploaded_files:
    indexed_names = {doc["name"] for doc in indexed_docs}
    newly_indexed = 0
    for uploaded_file in uploaded_files:
        if uploaded_file.name in indexed_names:
            st.info(f"{uploaded_file.name} is already indexed.")
            continue

        try:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                process_uploaded_file(uploaded_file)
            indexed_names.add(uploaded_file.name)
            newly_indexed += 1
            st.success(f"Indexed {uploaded_file.name} into the knowledge base.")
        except Exception as exc:
            st.error(f"Could not index {uploaded_file.name}: {exc}")

    # Rerun so the Knowledge Base sidebar (rendered above) reflects the freshly
    # indexed documents immediately instead of on the next interaction.
    if newly_indexed:
        st.rerun()

st.divider()

st.subheader("Ask a question")

sample_questions = load_sample_questions()
if sample_questions:
    def _use_sample():
        sel = st.session_state.get("sample_select", "")
        if sel and sel != "— Chọn —":
            st.session_state.question_input = sel

    with st.expander(f"📋 Câu hỏi mẫu ({len(sample_questions)} câu)"):
        st.selectbox(
            "Chọn một câu hỏi mẫu",
            options=["— Chọn —"] + sample_questions,
            index=0,
            key="sample_select",
            on_change=_use_sample,
        )

question = st.text_input(
    "Question",
    key="question_input",
    placeholder="Ví dụ: Giới hạn số giờ làm thêm trong ngày và trong tháng là bao nhiêu?",
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not indexed_docs:
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
                            f"**{chunk['source']}** (page {chunk.get('page', 'Unknown')}, "
                            f"similarity {chunk.get('similarity', '—')})"
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

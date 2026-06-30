# Simple RAG Requirement Assistant

## 1. Project Summary

**Project Name:** Simple RAG Requirement Assistant

**Project Type:** Beginner-Friendly Retrieval-Augmented Generation (RAG) Project

**Main Users:**

* QA Engineers
* Business Analysts (BA)
* Product Owners (PO)
* Technical Leads

The system allows users to upload requirement and project-related documents, search information, and ask questions using natural language.

The assistant retrieves relevant document sections and generates answers with source references.

---

## 2. Project Goal

Help QA and BA teams quickly search project documents and answer questions using uploaded content.

Example questions:

* What is the login validation rule?
* What are the requirements for UAT sign-off?
* Summarize the payment module requirements.
* What is the refund policy?
* Which requirement describes appointment booking?
* What security requirements exist for password reset?

---

## 3. Project Scope

### In Scope

* Upload requirement documents
* Extract text from documents
* Clean and chunk document content
* Create embeddings
* Store embeddings in a vector database
* Ask questions from uploaded documents
* Retrieve relevant chunks
* Generate answers using an LLM
* Show answers with source references
* Evaluate retrieval and answer quality

### Out of Scope

* Auto-generating full test case files
* Requirement-to-test-case traceability matrix
* User login and permission management
* Advanced dashboard
* Fine-tuning models
* Multi-agent workflow

---

## 4. Data Sources

| Document Type      | Purpose                                       |
| ------------------ | --------------------------------------------- |
| BRD                | Business requirements                         |
| FRD                | Functional requirements                       |
| SRS                | System requirements                           |
| Meeting Notes      | Requirement discussions                       |
| UAT Documents      | Acceptance criteria and sign-off requirements |
| Test Cases         | Existing QA assets                            |
| Release Notes      | Change tracking                               |
| API Specifications | Technical reference                           |

---

## 5. Architecture

```text
Requirement Documents
(BRD, FRD, SRS, Meeting Notes, Test Cases, Release Notes)
                    │
                    ▼
            Upload Documents
                    │
                    ▼
            Text Extraction
          PDF / DOCX / TXT / XLSX
                    │
                    ▼
             Text Cleaning
                    │
                    ▼
               Chunking
                    │
                    ▼
             Add Metadata
                    │
                    ▼
        Embedding Provider
 (Vietnamese / Ollama / BGE / OpenAI)
                    │
                    ▼
               ChromaDB
                    │
                    ▼
              Retriever
                    │
                    ▼
            LLM Provider
       (Ollama / OpenAI)
                    │
                    ▼
     Answer + Source Reference
```

---

## 6. Technology Stack

| Component       | Tool                                |
| --------------- | ----------------------------------- |
| Language        | Python                              |
| Frontend        | Streamlit                           |
| Backend         | FastAPI *(planned, not yet implemented)* |
| Parsing         | PyMuPDF, python-docx, openpyxl      |
| Embedding       | Vietnamese / Ollama / BGE / OpenAI  |
| LLM             | Ollama / OpenAI                     |
| Vector DB       | ChromaDB                            |
| Framework       | LangChain, sentence-transformers    |
| Version Control | GitHub                              |

Supported upload formats: **PDF, DOCX, TXT, XLSX**.

### Supported Embedding Providers

#### Vietnamese Embeddings (baseline)

```env
EMBED_PROVIDER=vietnamese
VI_EMBED_MODEL=AITeamVN/Vietnamese_Embedding
```

Advantages:

* Built on BGE-M3 (1024-dim), specialized for Vietnamese
* Best retrieval quality on Vietnamese corpora (the project's baseline model)
* Runs locally using `sentence-transformers`; no HTTP server or API key

Setup:

```bash
python scripts/setup_project.py --embed-provider vietnamese
```

---

#### Ollama Embeddings

```env
EMBED_PROVIDER=ollama
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Advantages:

* Free
* Local execution
* No API key required
* Good for local demo and beginner setup

Setup:

```bash
python scripts/setup_project.py --embed-provider ollama
```

---

#### BGE Embeddings

```env
EMBED_PROVIDER=bge
BGE_EMBED_MODEL=BAAI/bge-base-en-v1.5
```

Advantages:

* Strong retrieval performance
* Open source
* Runs locally using `sentence-transformers`
* No HTTP server required

Setup:

```bash
python scripts/setup_project.py --embed-provider bge
```

Important:

```text
BGE runs locally inside Python.
You do NOT need to open http://localhost:5001.
BGE_SERVER_URL is only needed if a separate BGE API server is built later.
```

---

#### OpenAI Embeddings

```env
EMBED_PROVIDER=openai
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_API_KEY=<your_api_key>
```

Advantages:

* High-quality embeddings
* Strong semantic understanding
* Suitable for production environments

Setup:

```bash
python scripts/setup_project.py --embed-provider openai
```

---

### Supported LLM Providers

#### Ollama LLM

```env
LLM_PROVIDER=ollama
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_URL=http://localhost:11434
```

Alternatives:

```text
mistral
qwen3:8b
llama3.1:8b
```

#### OpenAI LLM

```env
LLM_PROVIDER=openai
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=<your_api_key>
```

---

## 7. Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Developer Setup

Use when:

* Developing new features
* Processing real requirement documents
* Building vector databases from project documents

Default setup:

```bash
python scripts/setup_project.py
```

Setup with a specific embedding provider:

```bash
python scripts/setup_project.py --embed-provider vietnamese
python scripts/setup_project.py --embed-provider ollama
python scripts/setup_project.py --embed-provider bge
python scripts/setup_project.py --embed-provider openai
```

This will:

* Create project folders
* Setup selected embedding provider
* Pull or verify required models
* Build ChromaDB if `data/processed/chunks.json` exists

---

### Demo / Benchmark Setup

Use when:

* Learning the platform
* Onboarding new team members
* Running benchmark experiments
* Preparing demos and presentations

```bash
python scripts/setup_project.py --demo --embed-provider bge
```

This will additionally create:

```text
data/benchmark/
├── requirement_embedding_pairs.jsonl
├── requirement_chunks.jsonl
├── retrieval_eval.jsonl
└── llm_artifact_eval.jsonl
```

---

## 8. Benchmark Framework

This project includes 3 benchmark levels.

### Level 1 — Embedding Relevance Benchmark

Purpose:

Evaluate whether the embedding model understands requirement meaning.

Positive Requirement:

```text
Relevant requirement or document chunk
that SHOULD be retrieved.
```

Negative Requirement:

```text
Unrelated requirement or document chunk
that SHOULD NOT be retrieved.
```

PASS Condition:

```text
Similarity(Query, Positive Requirement)
>
Similarity(Query, Negative Requirement)
```

Run:

```bash
python scripts/benchmark_embeddings.py
```

Output:

```text
outputs/embeddings/<provider>/<model>/reports/embedding_pairwise_report.csv
```

---

### Level 2 — Retrieval Ranking Benchmark

Purpose:

Evaluate whether the retriever can find the correct requirement or document chunk.

Metrics:

* Recall@1
* Recall@3
* Recall@5
* MRR (Mean Reciprocal Rank)
* Latency

Example:

```text
Question:
What is the login lock rule?

Retrieved:

1. Login Requirement ✓
2. Payment Requirement
3. UAT Requirement
```

Run:

```bash
python scripts/benchmark_embeddings.py
```

Output:

```text
outputs/embeddings/<provider>/<model>/reports/retrieval_ranking_report.csv
outputs/embeddings/<provider>/<model>/reports/embedding_summary_report.md
```

---

### Level 3 — Answer Quality Benchmark

Purpose:

Evaluate whether the LLM generates correct answers using retrieved document content.

Metrics:

* Keyword Coverage
* Groundedness
* Latency

Example:

Question:

```text
What is the login lock rule?
```

Expected Answer:

```text
The system shall lock the user account after 5 failed login attempts.
```

Run:

```bash
python scripts/benchmark_llms.py
```

Output:

```text
outputs/llms/<provider>/<model>/reports/answer_quality_report.csv
outputs/llms/<provider>/<model>/reports/answer_quality_summary_report.md
```

---

### Level 4 — End-to-End Retrieval Evaluation

Purpose:

Unlike Levels 1–2 (which score an embedder in isolation on a generic sample set),
this evaluates the **full RAG pipeline** — chunking + embedding + ChromaDB — by
running queries against the **actual indexed corpus**.

Gold set: `eval/gold_match.jsonl` (match-based — a chunk is correct if its text
contains the `must_contain` marker, e.g. `"Điều 5."`, and matches `source`). This
makes the metric stable across re-chunking. `eval/BASELINE.md` and `eval/TUNING.md`
record the reference scores used for chunk-tuning experiments.

Run (set the provider and its collection to match):

```bash
EMBED_PROVIDER=vietnamese CHROMA_COLLECTION=qa_documents_vietnamese \
  python scripts/rag_eval.py score --gold eval/gold_match.jsonl --k 1 3 5 \
  --out outputs/comparison/rag_eval_vietnamese.json
```

Metrics: Recall@1, Recall@3, Recall@5, MRR (computed on the real corpus).

> Note: an English-centric embedder may look strong on the Level 1–2 generic set
> yet score poorly here on a Vietnamese corpus — only this end-to-end evaluation
> reveals real-world quality for your documents.

---

## 9. Benchmark Reports and Visualization

The benchmark framework generates both machine-readable reports and human-readable visualizations.

### Embedding Benchmark Outputs

```text
outputs/embeddings/<provider>/<model>/
├── reports/
│   ├── embedding_pairwise_report.csv
│   ├── retrieval_ranking_report.csv
│   └── embedding_summary_report.md
│
└── images/
    ├── embedding_accuracy.png
    ├── retrieval_recall.png
    ├── retrieval_mrr.png
    ├── embedding_latency.png
    └── benchmark_summary.png
```

Example: `outputs/embeddings/vietnamese/AITeamVN_Vietnamese_Embedding/reports/`

### LLM Benchmark Outputs

```text
outputs/llms/<provider>/<model>/
├── reports/
│   ├── answer_quality_report.csv
│   └── answer_quality_summary_report.md
│
└── images/
    ├── answer_keyword_coverage.png
    ├── answer_groundedness.png
    ├── answer_latency.png
    └── answer_quality_summary.png
```

### Example Embedding Summary

```text
<model> achieved Recall@3 = 95.24%
Average embedding latency = 0.0658s/document
Recommended for the Simple RAG Requirement Assistant
```

(Each summary is generated from real benchmark results in
`outputs/embeddings/<provider>/<model>/reports/embedding_summary_report.md`.)

### Comparing Embedding Providers

Benchmark each target provider (each writes its own per-model report):

```bash
python scripts/setup_project.py --embed-provider vietnamese
python scripts/benchmark_embeddings.py

python scripts/setup_project.py --embed-provider ollama
python scripts/benchmark_embeddings.py

python scripts/setup_project.py --embed-provider bge
python scripts/benchmark_embeddings.py
```

Then aggregate all per-model reports into side-by-side comparison tables:

```bash
python scripts/compare_models.py
# -> outputs/comparison/embedding_comparison.{csv,md}
# -> outputs/comparison/llm_comparison.{csv,md}
```

The Streamlit app also shows these comparisons (component benchmark + end-to-end
retrieval) in the **"Model quality comparison"** panel.

Compare:

* Embedding Relevance Accuracy
* Recall@1
* Recall@3
* Recall@5
* MRR
* Average Latency

---

## 10. Team Roles

### Technical Leader

Responsibilities:

* Project scope
* Architecture design
* Timeline management
* Integration review
* Final demo

### AI Engineer — Data

Responsibilities:

* Document collection
* Text extraction
* Text cleaning
* Chunking
* Metadata management
* Benchmark dataset preparation

### AI Engineer — Model

Responsibilities:

* Embedding model selection
* LLM selection
* Prompt engineering
* Retrieval configuration
* Benchmark analysis

### AI Engineer — Pipeline

Responsibilities:

* Streamlit UI
* ChromaDB integration
* Retriever integration
* LLM integration
* API integration
* Deployment support

### QA Engineer

Responsibilities:

* Retrieval testing
* Answer validation
* Benchmark validation
* End-to-end testing

---

## 11. Recommended Workflow

### New Developer

```bash
python scripts/setup_project.py
```

### New Team Member / Demo Setup

```bash
python scripts/setup_project.py --demo --embed-provider bge

python scripts/benchmark_embeddings.py

python scripts/benchmark_llms.py
```

### Process Real Documents

Place raw documents here:

```text
data/raw/
```

Then run:

```bash
python scripts/process_documents.py

python scripts/build_vectordb.py

python scripts/check_vectordb.py
```

### Run Application

```bash
streamlit run app.py
```

---

## 12. Default Configuration

```env
# Embedding Provider — options: vietnamese | ollama | bge | openai

EMBED_PROVIDER=vietnamese

VI_EMBED_MODEL=AITeamVN/Vietnamese_Embedding
OLLAMA_EMBED_MODEL=nomic-embed-text
BGE_EMBED_MODEL=BAAI/bge-base-en-v1.5
OPENAI_EMBED_MODEL=text-embedding-3-small

# LLM Provider — options: ollama | openai

LLM_PROVIDER=ollama

OLLAMA_LLM_MODEL=llama3.2
OPENAI_LLM_MODEL=gpt-4o-mini

# Services

OLLAMA_URL=http://localhost:11434

# ChromaDB
# The app and build_vectordb.py override this per provider as
# qa_documents_<EMBED_PROVIDER>, so each embedder keeps its own collection
# (vector dimensions never clash). The value below is only a fallback.

CHROMA_DB_PATH=./data/chroma
CHROMA_COLLECTION=qa_documents_vietnamese
```

---

## 13. Future Improvements

* User Story Generation
* Acceptance Criteria Generation
* Test Case Suggestion Generation
* Requirement Traceability Matrix (RTM)
* Duplicate Requirement Detection
* Requirement Change Impact Analysis
* Evaluation Dashboard
* Jira Integration
* TestRail Integration
* MCP Integration
* Multi-Agent Workflow

---

## 14. AIO 2026 Module 1 Objective

Demonstrate a working RAG application that:

* Processes requirement and project documents
* Retrieves relevant information
* Answers user questions
* Provides source references
* Evaluates retrieval quality
* Evaluates answer quality

using Vietnamese/Ollama/BGE/OpenAI embeddings, ChromaDB, LangChain, and Prompt Engineering.

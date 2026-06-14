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
             PDF / DOCX / TXT
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
    (Ollama / BGE / OpenAI)
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

| Component       | Tool                  |
| --------------- | --------------------- |
| Language        | Python                |
| Frontend        | Streamlit             |
| Backend         | FastAPI               |
| Parsing         | pypdf, python-docx    |
| Embedding       | Ollama / BGE / OpenAI |
| LLM             | Ollama / OpenAI       |
| Vector DB       | ChromaDB              |
| Framework       | LangChain             |
| Version Control | GitHub                |

### Supported Embedding Providers

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
gemma3
llama3.1
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
outputs/embeddings/reports/embedding_pairwise_report.csv
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
outputs/embeddings/reports/retrieval_ranking_report.csv
outputs/embeddings/reports/embedding_summary_report.md
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
outputs/llms/reports/answer_quality_report.csv
outputs/llms/reports/answer_quality_summary_report.md
```

---

## 9. Benchmark Reports and Visualization

The benchmark framework generates both machine-readable reports and human-readable visualizations.

### Embedding Benchmark Outputs

```text
outputs/embeddings/
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

### LLM Benchmark Outputs

```text
outputs/llms/
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
nomic-embed-text đạt Recall@3 = 91%
Thời gian embedding trung bình = 0.35s/document
Đề xuất sử dụng cho Simple RAG Requirement Assistant MVP
```

### Comparing Embedding Providers

To compare providers, update `.env` or run setup with the target provider:

```bash
python scripts/setup_project.py --embed-provider ollama
python scripts/benchmark_embeddings.py

python scripts/setup_project.py --embed-provider bge
python scripts/benchmark_embeddings.py

python scripts/setup_project.py --embed-provider openai
python scripts/benchmark_embeddings.py
```

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
# Embedding Provider

EMBED_PROVIDER=ollama

OLLAMA_EMBED_MODEL=nomic-embed-text
BGE_EMBED_MODEL=BAAI/bge-base-en-v1.5
OPENAI_EMBED_MODEL=text-embedding-3-small

# LLM Provider

LLM_PROVIDER=ollama

OLLAMA_LLM_MODEL=llama3.2
OPENAI_LLM_MODEL=gpt-4o-mini

# Services

OLLAMA_URL=http://localhost:11434

# Optional only if building separate BGE API server later
BGE_SERVER_URL=http://localhost:5001

# ChromaDB

CHROMA_DB_PATH=./data/chroma
CHROMA_COLLECTION=qa_documents
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

using Ollama, BGE/OpenAI embeddings, ChromaDB, LangChain, and Prompt Engineering.

```
```

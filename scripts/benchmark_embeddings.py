# scripts/benchmark_embeddings.py

"""
Benchmark Requirement Retrieval Quality.

Level 1:
Pairwise Requirement Retrieval Accuracy

Positive = relevant requirement
Negative = unrelated requirement

Level 2:
Retrieval Ranking Benchmark

Metrics:
- Recall@1
- Recall@3
- Recall@5
- MRR
- Latency

Outputs:
- CSV reports in outputs/embeddings/reports/
- PNG images in outputs/embeddings/images/
- Markdown summary in outputs/embeddings/reports/
"""

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from src.embeddings.factory import get_embedding_provider

load_dotenv()
def get_embedding_model_name() -> str:
    provider = os.getenv("EMBED_PROVIDER", "ollama").lower()

    if provider == "ollama":
        return os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    if provider == "bge":
        return os.getenv("BGE_EMBED_MODEL", "BAAI/bge-base-en-v1.5")

    if provider == "vietnamese":
        return os.getenv("VI_EMBED_MODEL", "AITeamVN/Vietnamese_Embedding")

    if provider == "openai":
        return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    return "unknown"
PAIR_FILE = Path("data/benchmark/requirement_embedding_pairs.jsonl")
CHUNK_FILE = Path("data/benchmark/requirement_chunks.jsonl")
RETRIEVAL_FILE = Path("data/benchmark/retrieval_eval.jsonl")
PROVIDER_NAME = os.getenv(
    "EMBED_PROVIDER",
    "ollama"
).lower()

MODEL_NAME = get_embedding_model_name()
MODEL_NAME_SAFE = (
    MODEL_NAME
    .replace("/", "_")
    .replace(":", "_")
    .replace(".", "_")
)

REPORT_DIR = Path(
    f"outputs/embeddings/{PROVIDER_NAME}/{MODEL_NAME_SAFE}/reports"
)

IMAGE_DIR = Path(
    f"outputs/embeddings/{PROVIDER_NAME}/{MODEL_NAME_SAFE}/images"
)

OUTPUT_PAIR = REPORT_DIR / "embedding_pairwise_report.csv"
OUTPUT_RETRIEVAL = REPORT_DIR / "retrieval_ranking_report.csv"
OUTPUT_SUMMARY_MD = REPORT_DIR / "embedding_summary_report.md"

IMAGE_ACCURACY = IMAGE_DIR / "embedding_accuracy.png"
IMAGE_RECALL = IMAGE_DIR / "retrieval_recall.png"
IMAGE_MRR = IMAGE_DIR / "retrieval_mrr.png"
IMAGE_LATENCY = IMAGE_DIR / "embedding_latency.png"
IMAGE_SUMMARY = IMAGE_DIR / "benchmark_summary.png"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"{path} is empty.")

    return rows


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def run_pairwise_benchmark(embedding_provider):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl(PAIR_FILE)

    rows = []
    passed = 0
    total_latency = 0.0

    for item in dataset:
        query = item["query"]
        positive = item["positive"]
        negative = item["negative"]

        start_time = time.time()

        query_vector = embedding_provider.embed_query(query)
        positive_vector = embedding_provider.embed_query(positive)
        negative_vector = embedding_provider.embed_query(negative)

        latency = time.time() - start_time
        total_latency += latency

        positive_score = cosine_similarity(query_vector, positive_vector)
        negative_score = cosine_similarity(query_vector, negative_vector)

        is_pass = positive_score > negative_score

        if is_pass:
            passed += 1

        rows.append(
            {
                "query": query,
                "positive_requirement": positive,
                "negative_requirement": negative,
                "positive_score": round(positive_score, 4),
                "negative_score": round(negative_score, 4),
                "margin": round(positive_score - negative_score, 4),
                "pass": is_pass,
                "latency_seconds": round(latency, 4),
                "embedding_dimension": len(query_vector),
            }
        )

    with open(OUTPUT_PAIR, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query",
                "positive_requirement",
                "negative_requirement",
                "positive_score",
                "negative_score",
                "margin",
                "pass",
                "latency_seconds",
                "embedding_dimension",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    accuracy = passed / len(dataset)
    avg_latency = total_latency / len(dataset)

    return {
        "total": len(dataset),
        "passed": passed,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "report": OUTPUT_PAIR,
    }


def run_retrieval_ranking_benchmark(embedding_provider):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_jsonl(CHUNK_FILE)
    eval_rows = load_jsonl(RETRIEVAL_FILE)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    chunk_texts = [chunk["text"] for chunk in chunks]

    print("Embedding requirement chunks...")

    chunk_start_time = time.time()
    chunk_vectors = embedding_provider.embed_documents(chunk_texts)
    chunk_embedding_latency = time.time() - chunk_start_time

    avg_document_embedding_latency = chunk_embedding_latency / len(chunk_texts)

    results = []

    recall_1 = 0
    recall_3 = 0
    recall_5 = 0
    reciprocal_rank_total = 0.0
    total_query_latency = 0.0

    for item in eval_rows:
        query = item["query"]
        expected_chunk_id = item["expected_chunk_id"]

        start_time = time.time()

        query_vector = embedding_provider.embed_query(query)

        scored_chunks = []

        for chunk_id, chunk_text, chunk_vector in zip(
            chunk_ids,
            chunk_texts,
            chunk_vectors,
        ):
            score = cosine_similarity(query_vector, chunk_vector)

            scored_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "score": score,
                }
            )

        scored_chunks.sort(
            key=lambda row: row["score"],
            reverse=True,
        )

        latency = time.time() - start_time
        total_query_latency += latency

        ranked_ids = [row["chunk_id"] for row in scored_chunks]

        rank = None

        if expected_chunk_id in ranked_ids:
            rank = ranked_ids.index(expected_chunk_id) + 1
            reciprocal_rank_total += 1 / rank

        if expected_chunk_id in ranked_ids[:1]:
            recall_1 += 1

        if expected_chunk_id in ranked_ids[:3]:
            recall_3 += 1

        if expected_chunk_id in ranked_ids[:5]:
            recall_5 += 1

        results.append(
            {
                "query": query,
                "expected_chunk_id": expected_chunk_id,
                "rank": rank,
                "reciprocal_rank": round(1 / rank, 4) if rank else 0,
                "top_1": ranked_ids[0] if len(ranked_ids) >= 1 else "",
                "top_3": ", ".join(ranked_ids[:3]),
                "top_5": ", ".join(ranked_ids[:5]),
                "latency_seconds": round(latency, 4),
            }
        )

    total = len(eval_rows)

    with open(OUTPUT_RETRIEVAL, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query",
                "expected_chunk_id",
                "rank",
                "reciprocal_rank",
                "top_1",
                "top_3",
                "top_5",
                "latency_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return {
        "total": total,
        "recall_at_1": recall_1 / total,
        "recall_at_3": recall_3 / total,
        "recall_at_5": recall_5 / total,
        "mrr": reciprocal_rank_total / total,
        "avg_query_latency": total_query_latency / total,
        "avg_document_embedding_latency": avg_document_embedding_latency,
        "report": OUTPUT_RETRIEVAL,
    }


def save_markdown_summary(
    provider_name: str,
    model_name: str,
    pair_result: dict,
    retrieval_result: dict,
):
    """
    Save a human-readable Markdown benchmark summary report.

    The report explains:
    - Accuracy
    - Recall@1
    - Recall@3
    - Recall@5
    - MRR
    - Latency
    - Recommendation for the Simple RAG Requirement Assistant
    """

    accuracy = pair_result["accuracy"]

    recall_at_1 = retrieval_result["recall_at_1"]
    recall_at_3 = retrieval_result["recall_at_3"]
    recall_at_5 = retrieval_result["recall_at_5"]
    mrr = retrieval_result["mrr"]

    avg_doc_latency = retrieval_result["avg_document_embedding_latency"]
    avg_query_latency = retrieval_result["avg_query_latency"]

    if recall_at_3 >= 0.90 and mrr >= 0.80:
        recommendation = (
            "Recommended for the Simple RAG Requirement Assistant."
        )
        reason = (
            "The model retrieves the correct requirement within the Top 3 "
            "results for most questions, and the correct result is usually "
            "ranked near the top."
        )

    elif recall_at_3 >= 0.75:
        recommendation = "Suitable for MVP and demo environments."
        reason = (
            "The model can retrieve relevant requirements reasonably well, "
            "but additional benchmarking against BGE or OpenAI embeddings "
            "is recommended."
        )

    else:
        recommendation = "Not recommended as the primary embedding model."
        reason = (
            "Retrieval performance is below the recommended threshold. "
            "Benchmarking with BGE or OpenAI embeddings is advised."
        )

    content = f"""# Embedding Benchmark Summary

## 1. Model Information

| Item | Value |
|------|-------|
| Provider | {provider_name} |
| Embedding Model | {model_name} |
| Pairwise Test Cases | {pair_result["total"]} |
| Retrieval Test Cases | {retrieval_result["total"]} |

---

## 2. Executive Summary

**{model_name}** achieved:

- Recall@1 = **{recall_at_1:.2%}**
- Recall@3 = **{recall_at_3:.2%}**
- Recall@5 = **{recall_at_5:.2%}**
- MRR = **{mrr:.4f}**
- Pairwise Retrieval Accuracy = **{accuracy:.2%}**
- Average Document Embedding Latency = **{avg_doc_latency:.4f}s/document**
- Average Query Retrieval Latency = **{avg_query_latency:.4f}s/query**

---

## 3. Recommendation

**{recommendation}**

{reason}

---

## 4. Metric Interpretation

### Accuracy

Accuracy checks whether the embedding model understands requirement meaning.

A test passes when:

```text
Similarity(Query, Relevant Requirement)
>
Similarity(Query, Unrelated Requirement)

## 5. Recommended Thresholds

| Metric | Recommended |
|--------|-------------|
| Accuracy | >= 90% |
| Recall@1 | >= 80% |
| Recall@3 | >= 90% |
| Recall@5 | >= 95% |
| MRR | >= 0.80 |
| Query Latency | < 0.1 sec |

---

## 6. Generated Files

### CSV Reports

- `{OUTPUT_PAIR}`
- `{OUTPUT_RETRIEVAL}`

These files contain detailed benchmark results for each test case.

### Visualization Images

- `{IMAGE_ACCURACY}`
- `{IMAGE_RECALL}`
- `{IMAGE_MRR}`
- `{IMAGE_LATENCY}`
- `{IMAGE_SUMMARY}`

These visualizations help compare retrieval quality and performance across embedding models.

---

## 7. Final Conclusion

### Benchmark Results

- Recall@1 = {recall_at_1:.2%}
- Recall@3 = {recall_at_3:.2%}
- Recall@5 = {recall_at_5:.2%}
- MRR = {mrr:.4f}
- Accuracy = {accuracy:.2%}
- Average Document Embedding Latency = {avg_doc_latency:.4f}s/document
- Average Query Retrieval Latency = {avg_query_latency:.4f}s/query

### Recommendation

**{recommendation}**

### Summary

{model_name} achieved Recall@3 = {recall_at_3:.2%}, indicating that the correct requirement appears within the Top 3 retrieved results for most benchmark queries.

The model achieved MRR = {mrr:.4f}, showing that relevant requirements are generally ranked near the top of the retrieval results.

Pairwise Retrieval Accuracy reached {accuracy:.2%}, demonstrating that the embedding model can distinguish relevant requirements from unrelated requirements effectively.

The average embedding latency was {avg_doc_latency:.4f}s/document and average retrieval latency was {avg_query_latency:.4f}s/query, providing acceptable performance for a Simple RAG Requirement Assistant.

**Final Verdict:** {recommendation}
"""

    OUTPUT_SUMMARY_MD.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_SUMMARY_MD.write_text(content,encoding="utf-8")


def save_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

    for index, value in enumerate(values):
        plt.text(
            index,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_summary_image(
    provider_name: str,
    model_name: str,
    pair_result: dict,
    retrieval_result: dict,
):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "Simple RAG Requirement Assistant",
        "",
        f"Embedding Provider: {provider_name}",
        f"Embedding Model: {model_name}",
        "",
        f"Embedding Accuracy: {pair_result['accuracy']:.2%}",
        f"Recall@1: {retrieval_result['recall_at_1']:.2%}",
        f"Recall@3: {retrieval_result['recall_at_3']:.2%}",
        f"Recall@5: {retrieval_result['recall_at_5']:.2%}",
        f"MRR: {retrieval_result['mrr']:.4f}",
        f"Avg Embedding Time: "
        f"{retrieval_result['avg_document_embedding_latency']:.4f}s/document",
    ]

    plt.figure(figsize=(9, 6))
    plt.axis("off")

    y = 0.95
    for line in lines:
        plt.text(
            0.05,
            y,
            line,
            fontsize=14 if line else 8,
            fontweight="bold" if line == lines[0] else "normal",
            transform=plt.gca().transAxes,
        )
        y -= 0.08

    plt.tight_layout()
    plt.savefig(IMAGE_SUMMARY)
    plt.close()


def generate_images(
    provider_name: str,
    model_name: str,
    pair_result: dict,
    retrieval_result: dict,
):
    save_bar_chart(
        labels=["Accuracy"],
        values=[pair_result["accuracy"] * 100],
        title=f"Embedding Relevance Accuracy - {model_name}",
        ylabel="Accuracy (%)",
        output_path=IMAGE_ACCURACY,
    )

    save_bar_chart(
        labels=["Recall@1", "Recall@3", "Recall@5"],
        values=[
            retrieval_result["recall_at_1"] * 100,
            retrieval_result["recall_at_3"] * 100,
            retrieval_result["recall_at_5"] * 100,
        ],
        title=f"Retrieval Recall - {model_name}",
        ylabel="Recall (%)",
        output_path=IMAGE_RECALL,
    )

    save_bar_chart(
        labels=["MRR"],
        values=[retrieval_result["mrr"]],
        title=f"Mean Reciprocal Rank - {model_name}",
        ylabel="MRR",
        output_path=IMAGE_MRR,
    )

    save_bar_chart(
        labels=["Document Embedding", "Query Retrieval"],
        values=[
            retrieval_result["avg_document_embedding_latency"],
            retrieval_result["avg_query_latency"],
        ],
        title=f"Average Latency - {model_name}",
        ylabel="Seconds",
        output_path=IMAGE_LATENCY,
    )

    save_summary_image(
        provider_name=provider_name,
        model_name=model_name,
        pair_result=pair_result,
        retrieval_result=retrieval_result,
    )


def main():
    embedding_provider = get_embedding_provider()
    provider_name = embedding_provider.__class__.__name__
    model_name = get_embedding_model_name()

    print("\nRequirement Assistant Embedding Benchmark")
    print("-----------------------------------------")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_name}")
    print("Positive = relevant requirement")
    print("Negative = unrelated requirement")

    pair_result = run_pairwise_benchmark(embedding_provider)
    retrieval_result = run_retrieval_ranking_benchmark(embedding_provider)

    generate_images(
        provider_name=provider_name,
        model_name=model_name,
        pair_result=pair_result,
        retrieval_result=retrieval_result,
    )

    save_markdown_summary(
        provider_name=provider_name,
        model_name=model_name,
        pair_result=pair_result,
        retrieval_result=retrieval_result,
    )

    print("\nLevel 1: Pairwise Requirement Retrieval")
    print("--------------------------------------")
    print(f"Total Cases: {pair_result['total']}")
    print(f"Passed: {pair_result['passed']}")
    print(f"Accuracy: {pair_result['accuracy']:.2%}")
    print(f"Average Latency: {pair_result['avg_latency']:.4f} sec")
    print(f"Report: {pair_result['report']}")

    print("\nLevel 2: Retrieval Ranking")
    print("--------------------------")
    print(f"Total Cases: {retrieval_result['total']}")
    print(f"Recall@1: {retrieval_result['recall_at_1']:.2%}")
    print(f"Recall@3: {retrieval_result['recall_at_3']:.2%}")
    print(f"Recall@5: {retrieval_result['recall_at_5']:.2%}")
    print(f"MRR: {retrieval_result['mrr']:.4f}")
    print(
        "Average document embedding time: "
        f"{retrieval_result['avg_document_embedding_latency']:.4f} sec/document"
    )
    print(
        "Average query retrieval time: "
        f"{retrieval_result['avg_query_latency']:.4f} sec/query"
    )
    print(f"Report: {retrieval_result['report']}")

    print("\nImage Outputs")
    print("-------------")
    print(f"Accuracy Chart: {IMAGE_ACCURACY}")
    print(f"Recall Chart: {IMAGE_RECALL}")
    print(f"MRR Chart: {IMAGE_MRR}")
    print(f"Latency Chart: {IMAGE_LATENCY}")
    print(f"Summary Dashboard: {IMAGE_SUMMARY}")

    print("\nMarkdown Report")
    print("---------------")
    print(f"Summary Report: {OUTPUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
# scripts/benchmark_llms.py

"""
Benchmark LLM answer quality for Simple RAG Requirement Assistant.

Level 3:
Answer Quality Benchmark

This benchmark checks whether the LLM can answer questions correctly
using provided requirement/document context.

Prompt templates are loaded from Markdown files:

src/templates/prompts/experiments/
├── qa_prompt_v1.md
├── qa_prompt_v2.md
└── qa_prompt_v3.md

Outputs:
- CSV reports in outputs/llms/reports/
- PNG images in outputs/llms/images/
- Markdown summary in outputs/llms/reports/
"""

import sys, os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import csv
import json
import time

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.llm.factory import get_llm_provider
from src.utils.prompt_loader import load_prompt

load_dotenv()

def get_llm_model_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        return os.getenv("OLLAMA_LLM_MODEL", "llama3.2")

    if provider == "openai":
        return os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    return "unknown"

EVAL_FILE = Path("data/benchmark/llm_artifact_eval.jsonl")

PROVIDER_NAME = os.getenv(
    "LLM_PROVIDER",
    "ollama",
).lower()

MODEL_NAME = get_llm_model_name()

MODEL_NAME_SAFE = (
    MODEL_NAME
    .replace("/", "_")
    .replace(":", "_")
    .replace(".", "_")
)

REPORT_DIR = Path(
    f"outputs/llms/{PROVIDER_NAME}/{MODEL_NAME_SAFE}/reports"
)

IMAGE_DIR = Path(
    f"outputs/llms/{PROVIDER_NAME}/{MODEL_NAME_SAFE}/images"
)

OUTPUT_FILE = REPORT_DIR / "answer_quality_report.csv"

IMAGE_KEYWORD_COVERAGE = IMAGE_DIR / "answer_keyword_coverage.png"
IMAGE_GROUNDEDNESS = IMAGE_DIR / "answer_groundedness.png"
IMAGE_LATENCY = IMAGE_DIR / "answer_latency.png"
IMAGE_SUMMARY = IMAGE_DIR / "answer_quality_summary.png"

PROMPT_TEMPLATE = "experiments/qa_prompt_v3.md"
OUTPUT_SUMMARY_MD = REPORT_DIR / "answer_quality_summary_report.md"


def save_markdown_summary(
    provider_name: str,
    model_name: str,
    avg_keyword_coverage: float,
    avg_groundedness: float,
    avg_latency: float,
    total_cases: int,
):
    """
    Save a human-readable Markdown benchmark summary report.

    The report explains:
    - Keyword Coverage
    - Groundedness
    - Latency
    - Recommendation for the Simple RAG Requirement Assistant
    """

    if avg_keyword_coverage >= 0.80 and avg_groundedness >= 0.80:
        recommendation = (
            "Recommended for the Simple RAG Requirement Assistant MVP."
        )
        reason = (
            "The model includes most expected requirement details and stays "
            "well grounded in the provided context."
        )

    elif avg_keyword_coverage >= 0.65 and avg_groundedness >= 0.70:
        recommendation = (
            "Suitable for demo usage, but prompt or model improvement is recommended."
        )
        reason = (
            "The model can generate usable answers, but some important details "
            "may be missing or the answer may not always stay fully grounded in the context."
        )

    else:
        recommendation = (
            "Not recommended as the primary LLM for this assistant yet."
        )
        reason = (
            "The model output quality is below the recommended threshold. "
            "Improve the prompt, try another prompt version, or benchmark another model."
        )

    content = f"""# LLM Answer Quality Benchmark Summary

## 1. Model Information

| Item | Value |
|------|-------|
| Provider | {provider_name} |
| LLM Model | {model_name} |
| Prompt Template | {PROMPT_TEMPLATE} |
| Benchmark Type | Answer Quality Benchmark |
| Total Test Cases | {total_cases} |

---

## 2. Executive Summary

**{model_name}** achieved:

- Keyword Coverage = **{avg_keyword_coverage:.2%}**
- Groundedness = **{avg_groundedness:.2%}**
- Average Answer Latency = **{avg_latency:.4f}s/question**

---

## 3. Recommendation

**{recommendation}**

{reason}

---

## 4. Metric Interpretation

### Keyword Coverage

Keyword Coverage measures whether the LLM answer includes the important expected details.

Result:

```text
Keyword Coverage = {avg_keyword_coverage:.2%}

## 5. Recommended Thresholds

| Metric | Recommended |
|--------|-------------|
| Keyword Coverage | >= 80% |
| Groundedness | >= 80% |
| Average Answer Latency | < 5 sec/question |

Interpretation:

- Keyword Coverage >= 80% indicates that most important requirement details are included in the generated answer.
- Groundedness >= 80% indicates that the answer remains closely aligned with the provided context.
- Lower latency provides a better user experience.

---

## 6. Generated Files

### CSV Report

- `{OUTPUT_FILE}`

This report contains detailed benchmark results for each test case, including:

- Question
- Context
- Expected Keywords
- Keyword Coverage
- Groundedness
- Latency
- Generated Answer

### Visualization Images

- `{IMAGE_KEYWORD_COVERAGE}`
- `{IMAGE_GROUNDEDNESS}`
- `{IMAGE_LATENCY}`
- `{IMAGE_SUMMARY}`

These visualizations help compare answer quality and response performance across different prompt templates and LLM models.

---

## 7. Final Conclusion

### Benchmark Results

- Keyword Coverage = {avg_keyword_coverage:.2%}
- Groundedness = {avg_groundedness:.2%}
- Average Answer Latency = {avg_latency:.4f}s/question

### Recommendation

**{recommendation}**

### Summary

{model_name} achieved Keyword Coverage of {avg_keyword_coverage:.2%}, indicating how often important requirement information appeared in generated answers.

The model achieved Groundedness of {avg_groundedness:.2%}, indicating how closely generated answers remained aligned with the provided requirement context.

The average response generation time was {avg_latency:.4f} seconds per question.

### Final Verdict

{recommendation}

Reason:

{reason}
"""

    OUTPUT_SUMMARY_MD.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_SUMMARY_MD.write_text(content,encoding="utf-8")

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


def build_prompt(question: str, context: str) -> str:
    template = load_prompt(PROMPT_TEMPLATE)

    return template.format(
        context=context,
        question=question,
    )


def calculate_keyword_coverage(
    output: str,
    expected_keywords: list[str],
) -> float:
    if not expected_keywords:
        return 0.0

    output_lower = output.lower()
    matched = 0

    for keyword in expected_keywords:
        if keyword.lower() in output_lower:
            matched += 1

    return matched / len(expected_keywords)


def calculate_groundedness(output: str, context: str) -> float:
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on",
        "for", "with", "is", "are", "be", "shall", "should",
        "can", "will", "this", "that", "it", "as", "by", "from",
        "using",
    }

    output_words = [
        word.strip(".,:;!?()[]{}").lower()
        for word in output.split()
    ]

    context_words = {
        word.strip(".,:;!?()[]{}").lower()
        for word in context.split()
    }

    meaningful_words = [
        word
        for word in output_words
        if word and word not in stopwords and len(word) > 2
    ]

    if not meaningful_words:
        return 0.0

    matched_words = [
        word
        for word in meaningful_words
        if word in context_words
    ]

    return len(matched_words) / len(meaningful_words)


def normalize_eval_item(item: dict) -> dict:
    if "question" in item and "context" in item:
        return {
            "question": item["question"],
            "context": item["context"],
            "expected_keywords": item.get("expected_keywords", []),
        }

    if "requirement" in item:
        return {
            "question": "What information is described in this requirement?",
            "context": item["requirement"],
            "expected_keywords": item.get("expected_keywords", []),
        }

    raise ValueError(f"Unsupported benchmark item format: {item}")


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
        plt.text(index, value, f"{value:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_summary_image(
    provider_name: str,
    avg_keyword_coverage: float,
    avg_groundedness: float,
    avg_latency: float,
    total_cases: int,
):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "Simple RAG Requirement Assistant",
        "",
        f"LLM Provider: {provider_name}",
        f"Prompt Template: {PROMPT_TEMPLATE}",
        "",
        f"Total Cases: {total_cases}",
        f"Keyword Coverage: {avg_keyword_coverage:.2%}",
        f"Groundedness: {avg_groundedness:.2%}",
        f"Average Latency: {avg_latency:.4f} sec",
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
    avg_keyword_coverage: float,
    avg_groundedness: float,
    avg_latency: float,
    total_cases: int,
):
    save_bar_chart(
        labels=["Keyword Coverage"],
        values=[avg_keyword_coverage * 100],
        title=f"Answer Keyword Coverage - {provider_name}",
        ylabel="Coverage (%)",
        output_path=IMAGE_KEYWORD_COVERAGE,
    )

    save_bar_chart(
        labels=["Groundedness"],
        values=[avg_groundedness * 100],
        title=f"Answer Groundedness - {provider_name}",
        ylabel="Groundedness (%)",
        output_path=IMAGE_GROUNDEDNESS,
    )

    save_bar_chart(
        labels=["Latency"],
        values=[avg_latency],
        title=f"Answer Generation Latency - {provider_name}",
        ylabel="Seconds",
        output_path=IMAGE_LATENCY,
    )

    save_summary_image(
        provider_name=provider_name,
        avg_keyword_coverage=avg_keyword_coverage,
        avg_groundedness=avg_groundedness,
        avg_latency=avg_latency,
        total_cases=total_cases,
    )

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    llm = get_llm_provider()
    provider_name = llm.__class__.__name__
    model_name = get_llm_model_name()

    dataset = load_jsonl(EVAL_FILE)

    rows = []
    total_keyword_coverage = 0.0
    total_groundedness = 0.0
    total_latency = 0.0

    print("\nSimple RAG Requirement Assistant LLM Benchmark")
    print("---------------------------------------------")
    print(f"Provider: {provider_name}")
    print(f"Prompt Template: {PROMPT_TEMPLATE}")

    for raw_item in dataset:
        item = normalize_eval_item(raw_item)

        question = item["question"]
        context = item["context"]
        expected_keywords = item["expected_keywords"]

        prompt = build_prompt(
            question=question,
            context=context,
        )

        start_time = time.time()
        output = llm.generate(prompt)
        latency = time.time() - start_time

        keyword_coverage = calculate_keyword_coverage(
            output=output,
            expected_keywords=expected_keywords,
        )

        groundedness = calculate_groundedness(
            output=output,
            context=context,
        )

        total_keyword_coverage += keyword_coverage
        total_groundedness += groundedness
        total_latency += latency

        rows.append(
            {
                "question": question,
                "context": context,
                "expected_keywords": ", ".join(expected_keywords),
                "prompt_template": PROMPT_TEMPLATE,
                "keyword_coverage": round(keyword_coverage, 4),
                "groundedness": round(groundedness, 4),
                "latency_seconds": round(latency, 4),
                "answer": output,
            }
        )

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "context",
                "expected_keywords",
                "prompt_template",
                "keyword_coverage",
                "groundedness",
                "latency_seconds",
                "answer",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    total_cases = len(dataset)
    avg_keyword_coverage = total_keyword_coverage / total_cases
    avg_groundedness = total_groundedness / total_cases
    avg_latency = total_latency / total_cases

    generate_images(
        provider_name=provider_name,
        avg_keyword_coverage=avg_keyword_coverage,
        avg_groundedness=avg_groundedness,
        avg_latency=avg_latency,
        total_cases=total_cases,
    )

    save_markdown_summary(
        provider_name=provider_name,
        model_name=model_name,
        avg_keyword_coverage=avg_keyword_coverage,
        avg_groundedness=avg_groundedness,
        avg_latency=avg_latency,
        total_cases=total_cases,
    )
    
    print("\nLevel 3: Answer Quality Benchmark")
    print("---------------------------------")
    print(f"Total Cases: {total_cases}")
    print(f"Average Keyword Coverage: {avg_keyword_coverage:.2%}")
    print(f"Average Groundedness: {avg_groundedness:.2%}")
    print(f"Average Latency: {avg_latency:.4f} sec")
    print(f"Report: {OUTPUT_FILE}")

    print("\nImage Outputs")
    print("-------------")
    print(f"Keyword Coverage Chart: {IMAGE_KEYWORD_COVERAGE}")
    print(f"Groundedness Chart: {IMAGE_GROUNDEDNESS}")
    print(f"Latency Chart: {IMAGE_LATENCY}")
    print(f"Summary Dashboard: {IMAGE_SUMMARY}")

    print("\nMarkdown Report")
    print("---------------")
    print(f"Summary Report: {OUTPUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
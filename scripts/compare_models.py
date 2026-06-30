# scripts/compare_models.py

"""
Aggregate per-model benchmark reports into side-by-side comparison tables.

All embedding and LLM benchmarks are run against the SAME fixed corpus and gold
set in data/benchmark/, so the per-model reports are directly comparable. This
script scans every report under outputs/ and produces:

- outputs/comparison/embedding_comparison.{csv,md}
- outputs/comparison/llm_comparison.{csv,md}

It only reads existing reports — generate them first with:

    EMBED_PROVIDER=<provider> python scripts/benchmark_embeddings.py
    LLM_PROVIDER=<provider>   python scripts/benchmark_llms.py
"""

import csv
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = ROOT_DIR / "outputs" / "comparison"

# Each kind maps to its report filename, the "model name" row label in the
# Model Information table, and the ordered metric columns to compare.
KINDS = {
    "embeddings": {
        "report_name": "embedding_summary_report.md",
        "model_label": "Embedding Model",
        "metrics": [
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "MRR",
            "Pairwise Retrieval Accuracy",
            "Average Document Embedding Latency",
            "Average Query Retrieval Latency",
        ],
    },
    "llms": {
        "report_name": "answer_quality_summary_report.md",
        "model_label": "LLM Model",
        "metrics": [
            "Keyword Coverage",
            "Groundedness",
            "Average Answer Latency",
        ],
    },
}


def _parse_report(path: Path, model_label: str) -> dict:
    """Return {provider, model, <metric>: value} parsed from one summary report."""
    provider = path.parts[-4]  # outputs/<kind>/<provider>/<model_safe>/reports/x.md
    model = path.parts[-3]
    metrics: dict[str, str] = {}
    in_summary = False

    for line in path.read_text(encoding="utf-8").splitlines():
        # Friendly model name + provider from the Model Information table.
        row = re.match(rf"\|\s*{re.escape(model_label)}\s*\|\s*(.+?)\s*\|", line)
        if row:
            model = row.group(1).strip()
        prov = re.match(r"\|\s*Provider\s*\|\s*(.+?)\s*\|", line)
        if prov:
            provider = prov.group(1).strip()

        if line.startswith("## 2. Executive Summary"):
            in_summary = True
            continue
        if in_summary and line.startswith("## "):
            in_summary = False
        if in_summary:
            m = re.match(r"\s*-\s*(.+?)\s*=\s*\*\*(.+?)\*\*", line)
            if m:
                metrics[m.group(1).strip()] = m.group(2).strip()

    return {"provider": provider, "model": model, **metrics}


def collect(kind: str) -> list[dict]:
    cfg = KINDS[kind]
    pattern = str(ROOT_DIR / "outputs" / kind / "*" / "*" / "reports" / cfg["report_name"])
    from glob import glob

    rows = [_parse_report(Path(p), cfg["model_label"]) for p in sorted(glob(pattern))]
    return rows


def write_comparison(kind: str, rows: list[dict]):
    cfg = KINDS[kind]
    columns = ["provider", "model"] + cfg["metrics"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / f"{kind.rstrip('s')}_comparison.csv"
    md_path = OUTPUT_DIR / f"{kind.rstrip('s')}_comparison.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [f"# {kind.capitalize()} Model Comparison", "", header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, md_path


def main():
    for kind in KINDS:
        rows = collect(kind)
        if not rows:
            print(f"[{kind}] No benchmark reports found — run the benchmark first.")
            continue
        csv_path, md_path = write_comparison(kind, rows)
        print(f"\n=== {kind.upper()} ({len(rows)} model(s)) ===")
        cols = ["model"] + KINDS[kind]["metrics"]
        print(" | ".join(cols))
        for row in rows:
            print(" | ".join(str(row.get(c, "")) for c in cols))
        print(f"-> {csv_path}")
        print(f"-> {md_path}")


if __name__ == "__main__":
    main()

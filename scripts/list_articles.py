"""list_articles.py - Liệt kê các 'Điều' có thật trong corpus đã chunk.

Mục đích: cung cấp danh mục Điều/nguồn làm gốc soạn gold set (tránh đoán sai
số Điều hoặc chọn Điều không có trong corpus -> luôn MISS).

Usage:
  python scripts/list_articles.py                      # in danh mục
  python scripts/list_articles.py --scaffold eval/gold_candidates.jsonl
                                                        # + ghi khung gold
"""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.json"
ART_RE = re.compile(r"Điều\s+(\d+)")


def load_chunks():
    if not CHUNKS.exists():
        raise FileNotFoundError(f"Không thấy {CHUNKS}. Chạy process_documents.py trước.")
    with open(CHUNKS, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks):
    """source -> {article_no: chunk_count}."""
    idx = defaultdict(lambda: defaultdict(int))
    for c in chunks:
        text = unicodedata.normalize("NFC", c.get("text", ""))
        src = c.get("source", "")
        # chỉ lấy số Điều xuất hiện ở ~80 ký tự đầu (phần heading)
        head = text[:80]
        m = ART_RE.search(head)
        if m:
            idx[src][int(m.group(1))] += 1
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaffold", default=None,
                    help="Ghi khung gold (1 dòng/Điều, query để trống) ra file .jsonl")
    args = ap.parse_args()

    chunks = load_chunks()
    idx = build_index(chunks)

    rows = []
    print(f"Tổng {len(chunks)} chunk; {len(idx)} nguồn\n")
    for src in sorted(idx):
        arts = sorted(idx[src])
        split = [a for a in arts if idx[src][a] > 1]
        print(f"== {src} ==")
        print(f"   Số Điều nhận diện được: {len(arts)}")
        print(f"   Danh sách: {arts}")
        if split:
            print(f"   ⚠ Điều bị cắt qua nhiều chunk (>1): {split}")
        print()
        for a in arts:
            rows.append({"source": src, "article": a, "n_chunks": idx[src][a]})

    if args.scaffold:
        out = Path(args.scaffold)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows, 1):
                # source khớp dạng không đuôi mở rộng để bền với .pdf/.docx
                src_marker = re.sub(r"\.[^.]+$", "", r["source"])
                entry = {
                    "id": f"q{i}",
                    "query": "",  # <-- ĐIỀN câu hỏi tự nhiên về Điều này
                    "must_contain": [f"Điều {r['article']}"],
                    "source": src_marker,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Đã ghi khung {len(rows)} dòng -> {out}")
        print("Mở file, điền 'query' cho các Điều bạn muốn dùng, xóa dòng không dùng.")


if __name__ == "__main__":
    main()

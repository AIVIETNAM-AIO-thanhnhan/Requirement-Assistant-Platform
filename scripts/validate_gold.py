"""validate_gold.py - Kiểm tra mỗi câu gold có chunk khớp trong corpus.

Bắt sớm: marker không tồn tại (luôn MISS) hoặc khớp quá nhiều (mơ hồ).
Usage: python scripts/validate_gold.py --gold eval/gold_match.jsonl
"""
import argparse
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.json"


def norm(s):
    return unicodedata.normalize("NFC", s or "").casefold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold_match.jsonl")
    args = ap.parse_args()

    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    idx = [(norm(c.get("text", "")), norm(c.get("source", ""))) for c in chunks]
    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]

    n_zero = 0
    for g in gold:
        marker = g["must_contain"][0]
        src = g.get("source", "")
        hits = sum(1 for t, s in idx
                   if all(norm(m) in t for m in g["must_contain"])
                   and norm(src) in s)
        flag = "OK" if hits >= 1 else "‼ ZERO"
        if hits == 0:
            n_zero += 1
        print(f"  [{flag}] {g['id']:>4} | {marker:<10} {src:<22} chunk khớp = {hits}")

    print(f"\nTổng {len(gold)} câu; số câu KHÔNG có chunk khớp (sẽ luôn MISS): {n_zero}")
    if n_zero:
        print("→ Sửa marker/source các câu ZERO trước khi chạy rag_eval.")


if __name__ == "__main__":
    main()

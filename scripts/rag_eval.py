"""rag_eval.py - Đo baseline Recall@K & MRR cho retrieval (match-based gold).

Usage
-----
python scripts/rag_eval.py score --gold eval/gold_match.jsonl --k 1 3 5

Yêu cầu: ChromaDB đã build (scripts/build_vectordb.py) và EMBED_PROVIDER trong
.env trỏ tới embedder dùng cho retrieval. Lưu ý: nên dùng embedder tiếng Việt
(vd AITeamVN/Vietnamese_Embedding) thì điểm số mới có ý nghĩa với corpus luật VN.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import load_gold, recall_at_k, reciprocal_rank
from src.evaluation.metrics import relevance_flags


def _flatten_chroma(res: dict) -> list[dict]:
    """Chroma trả về list lồng theo từng query; ở đây chỉ có 1 query."""
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    out = []
    for text, meta in zip(docs, metas):
        out.append({"text": text, "source": (meta or {}).get("source", "")})
    return out


def score(gold_path: str, ks: list[int]) -> dict:
    from src.embeddings.factory import get_embedding_provider
    from src.vectordb.chroma_store import ChromaStore

    gold = load_gold(gold_path)
    embedder = get_embedding_provider()
    store = ChromaStore()
    n = store.count()
    if n == 0:
        raise RuntimeError(
            "Collection rỗng. Chạy scripts/build_vectordb.py trước."
        )
    max_k = max(ks)

    per_query = []
    agg = {f"recall@{k}": 0.0 for k in ks}
    agg["mrr"] = 0.0

    for item in gold:
        q_emb = embedder.embed_query(item.query)
        res = store.query(query_embedding=q_emb, top_k=max_k)
        retrieved = _flatten_chroma(res)
        flags = relevance_flags(retrieved, item.as_dict())
        row = {"id": item.id, "query": item.query}
        for k in ks:
            r = recall_at_k(flags, k)
            row[f"recall@{k}"] = r
            agg[f"recall@{k}"] += r
        rr = reciprocal_rank(flags)
        row["rr"] = round(rr, 4)
        row["hit_rank"] = (flags.index(True) + 1) if any(flags) else None
        agg["mrr"] += rr
        per_query.append(row)

    m = len(gold)
    summary = {k: round(v / m, 4) for k, v in agg.items()}
    summary["n_queries"] = m
    summary["collection_count"] = n
    return {"summary": summary, "per_query": per_query}


def main():
    ap = argparse.ArgumentParser(description="RAG retrieval evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("score", help="Tính Recall@K & MRR")
    sp.add_argument("--gold", required=True, help="Đường dẫn gold_match.jsonl")
    sp.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    sp.add_argument("--out", default=None, help="Ghi kết quả JSON (tuỳ chọn)")
    args = ap.parse_args()

    if args.cmd == "score":
        result = score(args.gold, sorted(set(args.k)))
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        for row in result["per_query"]:
            mark = "OK " if row.get("hit_rank") else "MISS"
            print(f"  [{mark}] {row['id']}: rank={row['hit_rank']} "
                  f"rr={row['rr']} | {row['query'][:60]}")
        if args.out:
            Path(args.out).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nĐã ghi: {args.out}")


if __name__ == "__main__":
    main()

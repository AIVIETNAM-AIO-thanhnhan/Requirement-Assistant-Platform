"""metadata_rerank_ab.py - Đo tác động của re-rank theo document_type.

KHÔNG build lại DB: dùng lại collection baseline (qa_documents_vi). Mỗi câu lấy
top-N ứng viên, rồi thử 3 cách xếp hạng và đo Recall@K / MRR trên cùng gold:

  1. baseline   : giữ thứ tự theo khoảng cách (như rag_eval hiện tại)
  2. boost_luat : đưa mọi chunk document_type='luat' lên trước (heuristic triển
                  khai được, không cần biết câu hỏi nhắm văn bản nào)
  3. oracle     : đưa chunk đúng loại văn bản của ĐÁP ÁN lên trước (TRẦN lý thuyết
                  - chỉ để đo tiềm năng tối đa, KHÔNG triển khai được vì cần biết
                  trước nguồn đúng)

So sánh 3 cột cho biết: lever document_type đáng đầu tư đến đâu (oracle = trần),
và heuristic đơn giản (boost_luat) có lợi hay hại trên tập câu trộn nhiều nguồn.

Usage:
  python scripts/metadata_rerank_ab.py --gold eval/gold_match.jsonl --k 1 3 5 --topn 20
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import chromadb  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import os  # noqa: E402
from src.evaluation import load_gold, recall_at_k, reciprocal_rank  # noqa: E402
from src.evaluation.metrics import relevance_flags  # noqa: E402
from process_documents import guess_document_type  # noqa: E402

CHROMA_PATH = ROOT / "data" / "chroma"
COLLECTION = os.getenv("CHROMA_COLLECTION", "qa_documents_vi")


def rerank(cands, target_type):
    """Stable-sort: chunk có document_type==target_type lên trước, giữ nguyên
    thứ tự gốc trong từng nhóm. target_type=None -> giữ nguyên (baseline)."""
    if target_type is None:
        return cands
    return sorted(cands, key=lambda c: 0 if c.get("doctype") == target_type else 1)


def metrics_for(order, gold_item, ks):
    flags = relevance_flags(order, gold_item.as_dict())
    out = {f"recall@{k}": recall_at_k(flags, k) for k in ks}
    out["rr"] = reciprocal_rank(flags)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold_match.jsonl")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--topn", type=int, default=20, help="số ứng viên lấy về để rerank")
    ap.add_argument("--out", default="outputs/eval/rerank_ab.json")
    args = ap.parse_args()
    ks = sorted(set(args.k))

    from src.embeddings.factory import get_embedding_provider
    embedder = get_embedding_provider()

    gold = load_gold(args.gold)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_collection(COLLECTION)
    print(f"Collection '{COLLECTION}': {col.count()} chunk | gold {len(gold)} câu "
          f"| topN={args.topn}\n")

    schemes = ["baseline", "boost_luat", "oracle"]
    agg = {s: {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0} for s in schemes}

    for item in gold:
        q_emb = embedder.embed_query(item.query)
        res = col.query(query_embeddings=[q_emb], n_results=args.topn)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        cands = [{"text": t, "source": (m or {}).get("source", ""),
                  "doctype": (m or {}).get("document_type", "")}
                 for t, m in zip(docs, metas)]

        gold_doctype = guess_document_type(item.source or "")
        orders = {
            "baseline": rerank(cands, None),
            "boost_luat": rerank(cands, "luat"),
            "oracle": rerank(cands, gold_doctype),
        }
        for s in schemes:
            mm = metrics_for(orders[s], item, ks)
            for k in ks:
                agg[s][f"recall@{k}"] += mm[f"recall@{k}"]
            agg[s]["mrr"] += mm["rr"]

    m = len(gold)
    summary = {s: {k: round(v / m, 4) for k, v in agg[s].items()} for s in schemes}

    # In bảng
    cols = [f"recall@{k}" for k in ks] + ["mrr"]
    print(f"{'scheme':<12} " + " ".join(f"{c:>9}" for c in cols))
    for s in schemes:
        print(f"{s:<12} " + " ".join(f"{summary[s][c]:>9.4f}" for c in cols))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nĐã ghi: {args.out}")
    print("Đọc: oracle = trần nếu biết đúng loại văn bản; boost_luat = heuristic "
          "triển khai được; chênh oracle−baseline cho biết lever này đáng giá đến đâu.")


if __name__ == "__main__":
    main()

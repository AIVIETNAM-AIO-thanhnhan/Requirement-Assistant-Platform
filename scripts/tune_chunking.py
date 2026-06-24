"""tune_chunking.py - Grid-search CHUNK_SIZE / CHUNK_OVERLAP, đo trên gold 38 câu.

Quy trình mỗi cấu hình (size, overlap):
  1. clean text (1 lần, dùng lại) -> split_text(size, overlap)
  2. build collection tạm riêng (cùng metric L2 như baseline) + embed bằng đúng
     embedder hiện tại (get_embedding_provider)
  3. đo Recall@K / MRR trên cùng gold set
  4. xóa collection tạm

So kết quả với mốc trong eval/BASELINE.md (cấu hình baseline: size=1000, overlap=150).

Usage
-----
python scripts/tune_chunking.py --sizes 600 800 1000 1200 --overlaps 100 150 200 \
    --gold eval/gold_match.jsonl --k 1 3 5 --out outputs/eval/tune_results.json

Lưu ý: KHÔNG đụng collection baseline (qa_documents_vi). Embedder lấy từ .env hiện tại,
nên hãy chạy với đúng model đã khóa baseline (AITeamVN/Vietnamese_Embedding).
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

load_dotenv()  # nạp EMBED_PROVIDER, BGE_EMBED_MODEL, CHROMA_DB_PATH từ .env

from src.processing import clean_text, split_text  # noqa: E402
from src.evaluation import load_gold, recall_at_k, reciprocal_rank  # noqa: E402
from src.evaluation.metrics import relevance_flags  # noqa: E402

# Tái dùng logic gom nguồn của pipeline chính (không chạy main() vì có __main__ guard)
import process_documents as pd  # noqa: E402

CHROMA_PATH = pd.ROOT_DIR / "data" / "chroma"


def build_chunks(sources, size, overlap):
    """Tái dựng list chunk-dict đúng schema, với (size, overlap) cho trước."""
    out = []
    for name, (raw, filename) in sources.items():
        text = clean_text(raw)
        pieces = [c.strip() for c in split_text(text, size, overlap)
                  if len(c.strip()) >= pd.MIN_CHUNK_CHARS]
        doc_type = pd.guess_document_type(filename)
        for i, c in enumerate(pieces):
            out.append({
                "chunk_id": f"{name}__{i}",
                "text": c,
                "source": filename,
                "document_type": doc_type,
                "module": pd.MODULE,
            })
    return out


def eval_collection(collection, gold, query_embs, ks):
    max_k = max(ks)
    agg = {f"recall@{k}": 0.0 for k in ks}
    agg["mrr"] = 0.0
    for item, q_emb in zip(gold, query_embs):
        res = collection.query(query_embeddings=[q_emb], n_results=max_k)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        retrieved = [{"text": t, "source": (m or {}).get("source", "")}
                     for t, m in zip(docs, metas)]
        flags = relevance_flags(retrieved, item.as_dict())
        for k in ks:
            agg[f"recall@{k}"] += recall_at_k(flags, k)
        agg["mrr"] += reciprocal_rank(flags)
    m = len(gold)
    return {k: round(v / m, 4) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[600, 800, 1000, 1200])
    ap.add_argument("--overlaps", nargs="+", type=int, default=[100, 150, 200])
    ap.add_argument("--gold", default="eval/gold_match.jsonl")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--out", default="outputs/eval/tune_results.json")
    ap.add_argument("--keep", action="store_true", help="Giữ lại collection tạm")
    args = ap.parse_args()
    ks = sorted(set(args.k))

    from src.embeddings.factory import get_embedding_provider
    embedder = get_embedding_provider()

    gold = load_gold(args.gold)
    print(f"Gold: {len(gold)} câu | grid: {len(args.sizes)}x{len(args.overlaps)} "
          f"= {len(args.sizes) * len(args.overlaps)} cấu hình")

    # Query embeddings: chỉ phụ thuộc câu hỏi -> embed 1 lần, dùng lại mọi cấu hình
    print("Embedding câu hỏi (1 lần)...")
    query_embs = [embedder.embed_query(g.query) for g in gold]

    sources = pd.gather_sources()
    if not sources:
        print("Không có tài liệu trong data/raw/"); return

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    created, results = [], []
    try:
        for size in args.sizes:
            for overlap in args.overlaps:
                cname = f"tune_s{size}_o{overlap}"
                chunks = build_chunks(sources, size, overlap)
                # reset collection tạm nếu còn sót
                try:
                    client.delete_collection(cname)
                except Exception:
                    pass
                col = client.get_or_create_collection(name=cname)  # default L2
                created.append(cname)
                embs = embedder.embed_documents([c["text"] for c in chunks])
                col.add(
                    ids=[c["chunk_id"] for c in chunks],
                    documents=[c["text"] for c in chunks],
                    embeddings=embs,
                    metadatas=[{"source": c["source"],
                                "document_type": c["document_type"],
                                "module": c["module"]} for c in chunks],
                )
                metrics = eval_collection(col, gold, query_embs, ks)
                row = {"size": size, "overlap": overlap,
                       "n_chunks": len(chunks), **metrics}
                results.append(row)
                print(f"  size={size:>4} overlap={overlap:>3} | "
                      f"chunks={len(chunks):>4} | "
                      + " ".join(f"{k}={row[k]}" for k in
                                 [f'recall@{x}' for x in ks] + ['mrr']))
                if not args.keep:
                    client.delete_collection(cname)
                    created.remove(cname)
    finally:
        if not args.keep:
            for c in created:
                try:
                    client.delete_collection(c)
                except Exception:
                    pass

    # Sắp xếp theo mrr giảm dần để xem cấu hình tốt nhất
    results.sort(key=lambda r: (r["mrr"], r.get("recall@3", 0)), reverse=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nTop cấu hình (theo MRR): size={results[0]['size']} "
          f"overlap={results[0]['overlap']} -> mrr={results[0]['mrr']}")
    print(f"Baseline (size=1000, overlap=150): R@1=0.8158 R@3=0.9474 "
          f"R@5=0.9737 MRR=0.8781")
    print(f"Đã ghi: {out}")


if __name__ == "__main__":
    main()

"""metrics.py - Match-based relevance + Recall@K / MRR (thuần, dễ test).

Match-based: 1 chunk được coi là "đúng" nếu text của nó chứa TẤT CẢ marker
trong must_contain (và khớp source nếu gold có chỉ định source). Không phụ
thuộc chunk_id -> metric bất biến khi re-chunk.
"""

import unicodedata


def normalize(s: str) -> str:
    """NFC + casefold để so khớp ổn định với corpus đã NFC-normalize."""
    return unicodedata.normalize("NFC", s or "").casefold()


def chunk_is_relevant(chunk_text: str, chunk_source: str, gold: dict) -> bool:
    """True nếu chunk khớp gold: chứa mọi must_contain (+ khớp source nếu có)."""
    text_n = normalize(chunk_text)
    for marker in gold.get("must_contain", []):
        if normalize(marker) not in text_n:
            return False
    want_source = gold.get("source")
    if want_source:
        if normalize(want_source) not in normalize(chunk_source):
            return False
    return True


def relevance_flags(retrieved: list[dict], gold: dict) -> list[bool]:
    """retrieved: list các chunk theo thứ tự rank, mỗi chunk {text, source}."""
    return [
        chunk_is_relevant(c.get("text", ""), c.get("source", ""), gold)
        for c in retrieved
    ]


def recall_at_k(flags: list[bool], k: int) -> float:
    """Hit-rate@K: 1.0 nếu có ít nhất 1 chunk đúng trong top-k, ngược lại 0.0."""
    return 1.0 if any(flags[:k]) else 0.0


def reciprocal_rank(flags: list[bool]) -> float:
    """1/(thứ hạng chunk đúng đầu tiên); 0.0 nếu không có chunk đúng nào."""
    for i, hit in enumerate(flags):
        if hit:
            return 1.0 / (i + 1)
    return 0.0

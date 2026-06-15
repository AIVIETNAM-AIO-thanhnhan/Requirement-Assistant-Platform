"""chunker.py — Chia text thành chunk, ưu tiên ranh giới Điều/Chương."""

DEFAULT_SEPARATORS = ["\nĐiều ", "\nChương ", "\n\n", "\n", ". ", " ", ""]


def _merge(pieces, size, overlap):
    chunks, cur = [], ""
    for p in pieces:
        if not p:
            continue
        candidate = (cur + p) if cur else p
        if len(candidate) <= size:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            tail = chunks[-1][-overlap:] if (overlap and chunks) else ""
            cur = (tail + p) if (tail and len(tail + p) <= size) else p
    if cur:
        chunks.append(cur)
    return chunks


def split_text(text, size=1000, overlap=150, seps=None):
    """Chia đệ quy: thử cắt tại separator ưu tiên cao trước (Điều, Chương...)."""
    if seps is None:
        seps = DEFAULT_SEPARATORS
    if len(text) <= size:
        return [text] if text.strip() else []
    sep = next((s for s in seps if s and s in text), "")
    if sep == "":
        step = max(1, size - overlap)
        return [text[i:i + size] for i in range(0, len(text), step)]
    rest = seps[seps.index(sep) + 1:]
    parts = text.split(sep)
    parts = [(sep + p if i > 0 else p) for i, p in enumerate(parts)]
    pieces = []
    for p in parts:
        if len(p) <= size:
            pieces.append(p)
        else:
            pieces.extend(split_text(p, size, overlap, rest))
    return _merge(pieces, size, overlap)

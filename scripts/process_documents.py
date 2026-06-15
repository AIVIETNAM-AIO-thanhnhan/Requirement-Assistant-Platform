"""
process_documents.py
Purpose
-------
Trích xuất -> làm sạch -> chunk tài liệu trong data/raw/,
xuất data/processed/chunks.json đúng schema build_vectordb.py yêu cầu.

Schema mỗi chunk: chunk_id, text, source, page, document_type, module

Usage
-----
python scripts/process_documents.py
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.processing import parse, SUPPORTED, clean_text, split_text

# ---- Cấu hình ----
RAW_DIR = ROOT_DIR / "data" / "raw"
OCR_DIR = ROOT_DIR / "data" / "ocr"
OUT_FILE = ROOT_DIR / "data" / "processed" / "chunks.json"

MODULE = "lao_dong"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 30
MIN_PDF_TEXT = 50


def guess_document_type(filename: str) -> str:
    name = filename.lower()
    if "boluat" in name or "bo_luat" in name or "luat" in name:
        return "luat"
    if "nghidinh" in name or "nghi_dinh" in name:
        return "nghi_dinh"
    if "thongtu" in name or "thong_tu" in name:
        return "thong_tu"
    return "khac"


def gather_sources() -> dict:
    sources = {}
    if not RAW_DIR.exists():
        return sources
    for path in sorted(RAW_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        ext = path.suffix.lower()
        text = parse(path)
        if ext == ".pdf" and len(text.strip()) < MIN_PDF_TEXT:
            ocr_path = OCR_DIR / (path.stem + ".txt")
            if ocr_path.exists():
                text = ocr_path.read_text(encoding="utf-8")
            else:
                print(f"  ({path.name}: PDF scan, chưa có OCR -> bỏ qua)")
                continue
        name = path.stem
        if name in sources:
            name = f"{path.stem}_{ext.lstrip('.')}"
        sources[name] = (text, path.name)
    return sources


def main():
    sources = gather_sources()
    if not sources:
        print(f"Không tìm thấy tài liệu trong {RAW_DIR}")
        return
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for name, (raw, filename) in sources.items():
        text = clean_text(raw)
        chunks = [c for c in split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                  if len(c.strip()) >= MIN_CHUNK_CHARS]
        doc_type = guess_document_type(filename)
        print(f"  {name}: {len(text)} ký tự -> {len(chunks)} chunk ({doc_type})")
        for i, c in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{name}__{i}",
                "text": c.strip(),
                "source": filename,
                "page": "",
                "document_type": doc_type,
                "module": MODULE,
            })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\nTổng {len(all_chunks)} chunk -> {OUT_FILE.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

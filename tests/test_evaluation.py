"""test_evaluation.py - Kiểm tra module evaluation (metrics + load_gold).

Chạy offline, không cần model/ChromaDB:
    python tests/test_evaluation.py
"""

import sys
from pathlib import Path

# Đường dẫn tương đối: <root>/tests/test_evaluation.py -> ROOT = <root>
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import load_gold, recall_at_k, reciprocal_rank, chunk_is_relevant
from src.evaluation.metrics import relevance_flags, normalize

GOLD_PATH = ROOT / "eval" / "gold_match.jsonl"

# ── 1) chunk_is_relevant: must_contain (tất cả) + source ──────────────────────
gold = {"must_contain": ["Điều 105", "48 giờ"], "source": "boluat_laodong_2019"}
assert chunk_is_relevant("Điều 105. ... không quá 48 giờ trong 01 tuần",
                         "boluat_laodong_2019.pdf", gold) is True
assert chunk_is_relevant("Điều 105 nhưng thiếu marker kia",
                         "boluat_laodong_2019.pdf", gold) is False   # thiếu "48 giờ"
assert chunk_is_relevant("Điều 105 ... 48 giờ",
                         "nghidinh_145_2020.pdf", gold) is False     # sai source

# ── 2) Bất biến hoa/thường + dấu (NFC + casefold) ─────────────────────────────
assert chunk_is_relevant("điều 105 ... 48 GIỜ", "boluat_laodong_2019", gold) is True

# ── 3) recall_at_k ────────────────────────────────────────────────────────────
flags = [False, False, True, False, True]  # chunk đúng đầu tiên ở rank 3
assert recall_at_k(flags, 1) == 0.0
assert recall_at_k(flags, 3) == 1.0
assert recall_at_k(flags, 5) == 1.0
assert recall_at_k([False] * 5, 5) == 0.0

# ── 4) reciprocal_rank ────────────────────────────────────────────────────────
assert reciprocal_rank([False, False, True]) == 1 / 3
assert reciprocal_rank([True, False]) == 1.0
assert reciprocal_rank([False, False, False]) == 0.0

# ── 5) relevance_flags end-to-end trên list retrieved giả lập ─────────────────
retrieved = [
    {"text": "Điều 104 ...", "source": "boluat_laodong_2019"},
    {"text": "Điều 105 ... 48 giờ ...", "source": "boluat_laodong_2019"},
]
f = relevance_flags(retrieved, gold)
assert f == [False, True], f
assert reciprocal_rank(f) == 0.5

# ── 6) load_gold đọc đúng bộ gold thực tế ─────────────────────────────────────
# Đếm động số dòng hợp lệ (bỏ dòng trống/comment) để test không vỡ khi gold mở rộng.
expected_n = sum(
    1 for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
)
items = load_gold(GOLD_PATH)
assert len(items) == expected_n, f"load_gold trả {len(items)} != {expected_n} dòng hợp lệ"

# Mọi item phải hợp lệ về cấu trúc
for it in items:
    assert it.id, "thiếu id"
    assert it.query, f"{it.id}: thiếu query"
    assert isinstance(it.must_contain, list) and it.must_contain, f"{it.id}: must_contain rỗng"
    assert it.source, f"{it.id}: thiếu source"

by_id = {it.id: it for it in items}

# Anchor q1 (trạng thái gold hiện tại, đã verify)
assert by_id["q1"].must_contain == ["Điều 5."]
assert by_id["q1"].source == "BoLuatLaoDong_2019"

# Lock ngoại lệ đã biết: q11 dùng "Điều 99" KHÔNG có dấu chấm cuối
# (để tránh prefix-collision với Điều 990/991... — xem ghi chú gold format)
assert by_id["q11"].must_contain == ["Điều 99"]

print(f"ALL TESTS PASSED: {len(items)} gold items loaded; metrics verified.")

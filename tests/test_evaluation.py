import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/claude/repo")))

from src.evaluation import load_gold, recall_at_k, reciprocal_rank, chunk_is_relevant
from src.evaluation.metrics import relevance_flags, normalize

# 1) chunk_is_relevant: must_contain (tất cả) + source
gold = {"must_contain": ["Điều 105", "48 giờ"], "source": "boluat_laodong_2019"}
assert chunk_is_relevant("Điều 105. ... không quá 48 giờ trong 01 tuần",
                         "boluat_laodong_2019.pdf", gold) is True
assert chunk_is_relevant("Điều 105 nhưng thiếu marker kia",
                         "boluat_laodong_2019.pdf", gold) is False   # thiếu "48 giờ"
assert chunk_is_relevant("Điều 105 ... 48 giờ",
                         "nghidinh_145_2020.pdf", gold) is False     # sai source

# 2) Bất biến hoa/thường + dấu (NFC + casefold)
assert chunk_is_relevant("điều 105 ... 48 GIỜ", "boluat_laodong_2019", gold) is True

# 3) recall_at_k
flags = [False, False, True, False, True]  # chunk đúng đầu tiên ở rank 3
assert recall_at_k(flags, 1) == 0.0
assert recall_at_k(flags, 3) == 1.0
assert recall_at_k(flags, 5) == 1.0
assert recall_at_k([False]*5, 5) == 0.0

# 4) reciprocal_rank
assert reciprocal_rank([False, False, True]) == 1/3
assert reciprocal_rank([True, False]) == 1.0
assert reciprocal_rank([False, False, False]) == 0.0

# 5) relevance_flags end-to-end trên list retrieved giả lập
retrieved = [
    {"text": "Điều 104 ...", "source": "boluat_laodong_2019"},
    {"text": "Điều 105 ... 48 giờ ...", "source": "boluat_laodong_2019"},
]
f = relevance_flags(retrieved, gold)
assert f == [False, True], f
assert reciprocal_rank(f) == 0.5

# 6) load_gold đọc đúng template
items = load_gold("/home/claude/repo/eval/gold_match.jsonl")
assert len(items) == 6
assert items[0].id == "q1"
assert items[0].must_contain == ["Điều 105", "48 giờ"]
assert items[0].source == "boluat_laodong_2019"

print("ALL TESTS PASSED:", len(items), "gold items loaded; metrics verified.")

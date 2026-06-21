"""gold.py - Đọc gold set match-based (.jsonl)."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldItem:
    id: str
    query: str
    must_contain: list[str] = field(default_factory=list)
    source: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "must_contain": self.must_contain,
            "source": self.source,
        }


def load_gold(path: str | Path) -> list[GoldItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy gold file: {path}")
    items: list[GoldItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if "query" not in obj or "must_contain" not in obj:
                raise ValueError(
                    f"Dòng {line_no}: thiếu 'query' hoặc 'must_contain'."
                )
            items.append(
                GoldItem(
                    id=str(obj.get("id", line_no)),
                    query=obj["query"],
                    must_contain=list(obj["must_contain"]),
                    source=obj.get("source"),
                )
            )
    if not items:
        raise ValueError("Gold set rỗng.")
    return items

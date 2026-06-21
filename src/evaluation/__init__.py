"""Evaluation subpackage: match-based gold + Recall@K / MRR metrics."""
from src.evaluation.gold import load_gold, GoldItem
from src.evaluation.metrics import (
    normalize,
    chunk_is_relevant,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "load_gold",
    "GoldItem",
    "normalize",
    "chunk_is_relevant",
    "recall_at_k",
    "reciprocal_rank",
]

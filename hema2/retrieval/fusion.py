"""Multi-query retrieval with position decay + Reciprocal Rank Fusion (Eq. 4)."""
from __future__ import annotations
from collections.abc import Sequence


def rrf_multi_query(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings[i] = doc ids ranked for the query from the i-th most recent turn.
    score(d) = sum_i (1/(i+2)) * 1/(k + rank_i(d));  i starts at 0 for the most
    recent turn, so the decay weight is 1/(i+2) matching 1/(i+1) with 1-based i."""
    scores: dict[str, float] = {}
    for i, ranked in enumerate(rankings):
        w = 1.0 / (i + 2)
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])

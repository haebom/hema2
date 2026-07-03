"""Provenance-linked rehydration: expand a summary into verbatim source turns."""
from __future__ import annotations
from collections.abc import Callable
from hema2.core.types import Summary, Turn, MemoryItem


def rehydrate(summary: Summary, turn_lookup: dict[str, Turn],
              score: Callable[[Turn], float], token_count: Callable[[str], int],
              budget: int) -> list[MemoryItem]:
    """Return highest-scoring source turns of `summary` within `budget` tokens."""
    turns = sorted((turn_lookup[t] for t in summary.source_turns if t in turn_lookup),
                   key=score, reverse=True)
    out: list[MemoryItem] = []
    used = 0
    for t in turns:
        n = token_count(t.text)
        if used + n > budget:
            continue
        out.append(MemoryItem(text=t.text, tokens=n, channel="rehydration",
                              source_turns={t.turn_id}))
        used += n
    return out

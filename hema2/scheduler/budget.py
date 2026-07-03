"""Multi-channel token budget allocation with carryover (Eq. 3 in the paper)."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from hema2.core.types import MemoryItem

Channel = Callable[[int], list[MemoryItem]]  # budget -> items (greedy, within budget)


def compose(total_budget: int, ratios: Sequence[float], channels: Sequence[Channel]) -> list[MemoryItem]:
    """Fill channels in order; unused allocation carries over: B_c = r_c*B + L_{c-1}."""
    assert len(ratios) == len(channels)
    assert abs(sum(ratios) - 1.0) < 1e-6, "channel ratios must sum to 1"
    selected: list[MemoryItem] = []
    leftover = 0.0
    for r, ch in zip(ratios, channels):
        budget_c = int(r * total_budget + leftover)
        items = ch(budget_c)
        used = sum(i.tokens for i in items)
        assert used <= budget_c, "channel exceeded its budget"
        selected.extend(items)
        leftover = budget_c - used
    return selected

"""Consolidation trigger policies (turn-based, capacity-based, hybrid)."""
from __future__ import annotations


class TurnTrigger:
    def __init__(self, every: int = 100) -> None:
        self.every = every

    def should_fire(self, turn_index: int, block_count: int) -> bool:
        return turn_index > 0 and turn_index % self.every == 0


class CapacityTrigger:
    def __init__(self, max_blocks: int = 4) -> None:
        self.max_blocks = max_blocks

    def should_fire(self, turn_index: int, block_count: int) -> bool:
        return block_count >= self.max_blocks


class HybridTrigger:
    def __init__(self, every: int = 100, max_blocks: int = 4) -> None:
        self.turn = TurnTrigger(every)
        self.cap = CapacityTrigger(max_blocks)

    def should_fire(self, turn_index: int, block_count: int) -> bool:
        return self.turn.should_fire(turn_index, block_count) or \
               self.cap.should_fire(turn_index, block_count)

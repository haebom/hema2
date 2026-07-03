"""Semantic Fact Store: ADD / MERGE / INVALIDATE with temporal queryability."""
from __future__ import annotations
from hema2.core.types import Fact


class SemanticFactStore:
    def __init__(self) -> None:
        self._facts: list[Fact] = []

    def add(self, fact: Fact) -> None:
        """ADD: insert; if a valid fact with same (subject, attribute) but a
        conflicting value exists, INVALIDATE it first (newer fact wins)."""
        for f in self._facts:
            if f.is_valid and f.subject == fact.subject and f.attribute == fact.attribute:
                if f.value != fact.value:
                    f.t_invalid = fact.t_start  # mark, never delete
                else:
                    # MERGE: same assertion re-stated; extend provenance only
                    f.source_turns |= fact.source_turns
                    return
        self._facts.append(fact)

    def invalidate(self, subject: str, attribute: str, at_turn: int) -> int:
        n = 0
        for f in self._facts:
            if f.is_valid and f.subject == subject and f.attribute == attribute:
                f.t_invalid = at_turn
                n += 1
        return n

    def query_valid(self, subject: str | None = None) -> list[Fact]:
        """Present-tense queries read only valid facts."""
        return [f for f in self._facts if f.is_valid
                and (subject is None or f.subject == subject)]

    def query_at(self, turn: int, subject: str | None = None) -> list[Fact]:
        """Temporal queries may read invalidated facts that were valid at `turn`."""
        return [f for f in self._facts
                if f.t_start <= turn and (f.t_invalid is None or f.t_invalid > turn)
                and (subject is None or f.subject == subject)]

    def invalidate_by_source(self, turn_ids: set[str], at_turn: int) -> int:
        """Edit consistency: one-pass invalidation of facts derived from edited turns."""
        n = 0
        for f in self._facts:
            if f.is_valid and f.source_turns & turn_ids:
                f.t_invalid = at_turn
                n += 1
        return n

    def __len__(self) -> int:
        return len(self._facts)

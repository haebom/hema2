"""Core data models for HEMA-2."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Turn:
    turn_id: str
    index: int
    role: str  # "user" | "assistant"
    text: str


@dataclass
class Fact:
    """Five-tuple fact with history-preserving invalidation (Eq. 2 in the paper)."""
    subject: str
    attribute: str
    value: str
    t_start: int
    t_invalid: int | None = None
    source_turns: set[str] = field(default_factory=set)

    @property
    def is_valid(self) -> bool:
        return self.t_invalid is None


@dataclass
class Summary:
    text: str
    source_turns: set[str] = field(default_factory=set)  # provenance
    pinned: bool = False


@dataclass
class MemoryItem:
    """A candidate prompt block produced by a channel."""
    text: str
    tokens: int
    channel: str
    source_turns: set[str] = field(default_factory=set)

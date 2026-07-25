"""Provider-neutral interfaces for cross-model evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict[str, str]]
    temperature: float = 0.0
    max_tokens: int = 256
    seed: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    provider: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict | None = None


class ChatModel(Protocol):
    provider: str
    model: str

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Generate one deterministic chat completion when supported."""
        ...

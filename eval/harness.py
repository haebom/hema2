"""Versioned evaluation harness for fair cross-model HEMA-2 comparisons.

The harness is deliberately separated from model adapters and memory systems.
Every result records the harness version, prompt policy, model profile, provider
model ID, memory condition, and seed so later harness evolution can be audited
instead of silently changing benchmark meaning.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

from hema2.core.types import Turn
from hema2.models.base import ModelRequest

HARNESS_VERSION = "hema2-e1-v1"
SYSTEM_PROMPT = (
    "You are evaluating conversational memory. Answer only from the supplied "
    "memory and conversation. For current-state questions, prefer the latest "
    "valid fact. Mention an older value only when the question asks about history. "
    "If the answer is not supported, say UNKNOWN."
)


@dataclass(frozen=True)
class MemoryCase:
    condition: str
    attribute: str
    latest: str
    stale: set[str]
    memory_text: str
    question: str
    turn_count: int


@dataclass(frozen=True)
class HarnessManifest:
    version: str = HARNESS_VERSION
    system_prompt_sha256: str = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    temperature: float = 0.0
    max_tokens: int = 96
    seed: int = 0
    adaptation: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


def render_turns(turns: Iterable[Turn]) -> str:
    return "\n".join(f"[{t.index}] {t.role}: {t.text}" for t in turns)


def make_request(case: MemoryCase, manifest: HarnessManifest, guide: str | None = None) -> ModelRequest:
    guide_block = f"\nEvaluation guide:\n{guide}\n" if guide else ""
    user = (
        f"Memory condition: {case.condition}\n"
        f"Conversation length: {case.turn_count} turns\n"
        f"Memory:\n{case.memory_text or '(none)'}\n"
        f"{guide_block}"
        f"Question: {case.question}\n"
        "Answer in one short sentence."
    )
    return ModelRequest(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=manifest.temperature,
        max_tokens=manifest.max_tokens,
        seed=manifest.seed,
        metadata={"harness_version": manifest.version, "condition": case.condition},
    )


def save_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

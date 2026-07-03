"""E1 generator: dialogues in which facts are revised N times.

Emits (turns, probes): each probe asks for the latest value of an attribute
and lists all superseded values, enabling stale-fact-rate scoring.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from hema2.core.types import Turn

_SUBJECTS = ["user"]
_ATTRS = {
    "city": ["Seoul", "Busan", "Daejeon", "Incheon", "Gwangju"],
    "employer": ["Acme", "Globex", "Initech", "Umbrella", "Hooli"],
    "hobby": ["climbing", "painting", "chess", "running", "pottery"],
}
_TEMPLATE = "By the way, my {attr} is now {value}."
_FILLER = ["Tell me something interesting.", "What do you think about that?",
           "Let's continue our discussion.", "Here is another question for you."]


@dataclass
class Probe:
    attribute: str
    latest: str
    stale: set[str]
    asked_at: int


def generate(n_updates: int = 3, filler_between: int = 20, seed: int = 0):
    rng = random.Random(seed)
    turns: list[Turn] = []
    probes: list[Probe] = []
    idx = 0

    def emit(role: str, text: str) -> None:
        nonlocal idx
        turns.append(Turn(turn_id=f"t{idx}", index=idx, role=role, text=text))
        idx += 1

    for attr, values in _ATTRS.items():
        seq = rng.sample(values, k=min(n_updates + 1, len(values)))
        for v in seq:
            emit("user", _TEMPLATE.format(attr=attr, value=v))
            emit("assistant", "Noted.")
            for _ in range(filler_between):
                emit("user", rng.choice(_FILLER))
                emit("assistant", "Sure, let us talk about that.")
        probes.append(Probe(attribute=attr, latest=seq[-1],
                            stale=set(seq[:-1]), asked_at=idx))
        emit("user", f"Remind me, what is my current {attr}?")
    return turns, probes

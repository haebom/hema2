"""Answer-level metrics that distinguish correct history from stale assertions."""
from __future__ import annotations

from collections import Counter


def classify_current_answer(answer: str, latest: str, stale: set[str]) -> str:
    """Classify a response to a current-state probe.

    This intentionally exposes mixed answers rather than counting any stale
    string mention as failure.  A response that names both the latest and an old
    value is reported separately for later human or judge-model inspection.
    """
    normalized = answer.casefold()
    has_latest = latest.casefold() in normalized
    stale_hits = {value for value in stale if value.casefold() in normalized}
    unknown = "unknown" in normalized or "not enough" in normalized or "cannot determine" in normalized

    if has_latest and stale_hits:
        return "mixed_latest_and_stale"
    if has_latest:
        return "correct_latest"
    if stale_hits:
        return "stale_only"
    if unknown:
        return "abstained"
    return "unsupported_other"


def aggregate_classifications(labels: list[str]) -> dict[str, float | int]:
    counts = Counter(labels)
    total = len(labels)
    result: dict[str, float | int] = {"n": total}
    for label in (
        "correct_latest",
        "mixed_latest_and_stale",
        "stale_only",
        "abstained",
        "unsupported_other",
    ):
        result[label] = counts[label]
        result[f"{label}_rate"] = counts[label] / total if total else 0.0
    result["update_accuracy"] = counts["correct_latest"] / total if total else 0.0
    result["strict_stale_rate"] = counts["stale_only"] / total if total else 0.0
    result["mixed_rate"] = counts["mixed_latest_and_stale"] / total if total else 0.0
    return result

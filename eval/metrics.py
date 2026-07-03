"""Evaluation metrics defined in the HEMA-2 protocol (Sec. V)."""
from __future__ import annotations
from collections.abc import Sequence


def stale_fact_rate(answers: Sequence[str], stale_values: Sequence[set[str]]) -> float:
    """Fraction of answers asserting a superseded value."""
    assert len(answers) == len(stale_values)
    if not answers:
        return 0.0
    hits = sum(1 for a, stale in zip(answers, stale_values)
               if any(s.lower() in a.lower() for s in stale))
    return hits / len(answers)


def update_accuracy(answers: Sequence[str], latest_values: Sequence[str]) -> float:
    """Fraction of answers containing the latest correct value."""
    assert len(answers) == len(latest_values)
    if not answers:
        return 0.0
    hits = sum(1 for a, v in zip(answers, latest_values) if v.lower() in a.lower())
    return hits / len(answers)


def abstention_f1(predicted_abstain: Sequence[bool], should_abstain: Sequence[bool]) -> float:
    tp = sum(p and s for p, s in zip(predicted_abstain, should_abstain))
    fp = sum(p and not s for p, s in zip(predicted_abstain, should_abstain))
    fn = sum((not p) and s for p, s in zip(predicted_abstain, should_abstain))
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def degradation_slope(turn_points: Sequence[int], recalls: Sequence[float]) -> float:
    """Least-squares slope of recall vs. turns, in pp lost per 1K turns (positive = loss)."""
    assert len(turn_points) == len(recalls) and len(recalls) >= 2
    n = len(turn_points)
    xs = [t / 1000.0 for t in turn_points]
    mx, my = sum(xs) / n, sum(recalls) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, recalls))
    den = sum((x - mx) ** 2 for x in xs)
    return -(num / den) * 100.0

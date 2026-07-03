"""Minimal test suite (plain asserts; run: python tests/test_all.py)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))

from hema2.core.types import Fact, Summary, Turn, MemoryItem
from hema2.memory.fact_store import SemanticFactStore
from hema2.scheduler.budget import compose
from hema2.retrieval.fusion import rrf_multi_query
from hema2.retrieval.rehydration import rehydrate
from hema2.consolidation.triggers import TurnTrigger, CapacityTrigger, HybridTrigger
from eval.metrics import stale_fact_rate, update_accuracy, abstention_f1, degradation_slope
from eval.synthetic.update_stress import generate


def test_fact_store():
    s = SemanticFactStore()
    s.add(Fact("user", "city", "Seoul", t_start=1, source_turns={"t1"}))
    s.add(Fact("user", "city", "Busan", t_start=50, source_turns={"t50"}))
    valid = s.query_valid("user")
    assert len(valid) == 1 and valid[0].value == "Busan"          # latest wins
    at_10 = s.query_at(10, "user")
    assert len(at_10) == 1 and at_10[0].value == "Seoul"          # temporal query reads history
    assert len(s) == 2                                            # nothing deleted
    s.add(Fact("user", "city", "Busan", t_start=60, source_turns={"t60"}))
    assert len(s) == 2                                            # MERGE, not duplicate
    n = s.invalidate_by_source({"t50", "t60"}, at_turn=70)
    assert n == 1 and len(s.query_valid("user")) == 0             # edit consistency


def test_scheduler_carryover():
    def ch(items):
        def fill(budget):
            out, used = [], 0
            for text, tok in items:
                if used + tok <= budget:
                    out.append(MemoryItem(text, tok, "x")); used += tok
            return out
        return fill
    # total 100, ratios .5/.5 ; channel 1 uses only 10 -> channel 2 gets 40 carryover
    sel = compose(100, [0.5, 0.5], [ch([("a", 10)]), ch([("b", 80), ("c", 10)])])
    assert [i.text for i in sel] == ["a", "b", "c"]
    assert sum(i.tokens for i in sel) == 100


def test_rrf():
    ranked = rrf_multi_query([["d1", "d2"], ["d2", "d3"]], k=60)
    order = [d for d, _ in ranked]
    assert order[0] == "d2"          # appears in both queries -> fused to top
    assert order == ["d2", "d1", "d3"]  # then recency-weighted d1 over d3


def test_rehydration_budget():
    turns = {f"t{i}": Turn(f"t{i}", i, "user", "x" * 10) for i in range(5)}
    summ = Summary("gist", source_turns=set(turns))
    out = rehydrate(summ, turns, score=lambda t: t.index,
                    token_count=lambda s: 10, budget=25)
    assert len(out) == 2 and all(i.tokens == 10 for i in out)
    assert out[0].source_turns == {"t4"}  # highest scoring first


def test_triggers():
    assert TurnTrigger(100).should_fire(200, 1)
    assert not TurnTrigger(100).should_fire(150, 99)
    assert CapacityTrigger(4).should_fire(3, 4)
    assert HybridTrigger(100, 4).should_fire(1, 5)


def test_metrics():
    assert stale_fact_rate(["I live in Seoul"], [{"Seoul"}]) == 1.0
    assert update_accuracy(["You moved to Busan"], ["Busan"]) == 1.0
    assert abstention_f1([True, False], [True, False]) == 1.0
    slope = degradation_slope([0, 1000, 2000], [0.9, 0.8, 0.7])
    assert abs(slope - 10.0) < 1e-6   # 10 pp lost per 1K turns


def test_e1_generator():
    turns, probes = generate(n_updates=2, filler_between=2, seed=1)
    assert len(probes) == 3
    for p in probes:
        assert p.latest not in p.stale and len(p.stale) == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")

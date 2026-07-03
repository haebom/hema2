"""E2 (memory-layer) long-horizon degradation simulation.

Isolates the storage/scheduling layer from the language model: final answer
accuracy is upper-bounded by whether the latest correct fact appears in the
composed prompt, so we measure prompt-level update recall and stale-fact
contamination under idealized components deliberately favorable to the
baseline (perfect topic retrieval, oracle reader).

Systems:
  dual-sim  : dual-memory baseline, similarity retrieval (score ties among
              same-topic assertions -> uniform sample of top-k)
  dual-rec  : dual-memory baseline with recency-ranked retrieval (stronger)
  tri       : tri-memory (HEMA-2) fact store with INVALIDATE; extractor
              error rate eps per update event

Usage: python eval/simulate.py --turns 10000 --seeds 10 --out results.json
"""
from __future__ import annotations
import argparse, json, random, statistics


def run_seed(seed: int, turns: int, n_attrs: int = 40, p_update: float = 1/400,
             top_k: int = 5, summary_cap: int = 15, eps: float = 0.05,
             checkpoints=(100, 500, 1000, 2000, 5000, 10000)):
    rng = random.Random(seed)
    # assertion history per attribute: list of turn indices (values implicit)
    hist: list[list[int]] = [[] for _ in range(n_attrs)]
    captured: list[bool] = [True] * n_attrs   # tri-memory: latest update captured?
    for a in range(n_attrs):                  # initial assertions in first 100 turns
        hist[a].append(rng.randrange(0, 100))
    results = {}
    cps = [c for c in checkpoints if c <= turns]
    for t in range(100, turns + 1):
        for a in range(n_attrs):
            if rng.random() < p_update:
                hist[a].append(t)
                captured[a] = rng.random() > eps   # extractor misses w.p. eps
        if t in cps:
            m = {k: 0.0 for k in ("dsim_rec", "dsim_stale", "drec_rec",
                                   "drec_stale", "tri_rec", "tri_stale")}
            # summary hit: attribute among summary_cap most recently updated
            order = sorted(range(n_attrs), key=lambda a: -hist[a][-1])
            in_summary = set(order[:summary_cap])
            for a in range(n_attrs):
                n = len(hist[a])
                s_hit = a in in_summary
                # similarity retrieval: uniform top_k among n tied assertions
                if n <= top_k:
                    v_hit, v_stale = True, n >= 2
                else:
                    v_hit = rng.random() < top_k / n
                    v_stale = True
                m["dsim_rec"] += s_hit or v_hit
                m["dsim_stale"] += v_stale
                # recency retrieval: latest always in, stale iff n>=2
                m["drec_rec"] += 1.0
                m["drec_stale"] += n >= 2
                # tri-memory: single valid fact; stale answer iff last update missed
                m["tri_rec"] += captured[a]
                m["tri_stale"] += not captured[a]
            results[t] = {k: v / n_attrs for k, v in m.items()}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="e2_sim_results.json")
    args = ap.parse_args()
    per_seed = [run_seed(s, args.turns) for s in range(args.seeds)]
    cps = sorted(per_seed[0].keys())
    agg = {}
    for t in cps:
        agg[t] = {}
        for k in per_seed[0][t]:
            vals = [ps[t][k] for ps in per_seed]
            mean = statistics.mean(vals)
            ci = 1.96 * statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
            agg[t][k] = {"mean": round(mean, 4), "ci95": round(ci, 4)}
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=1)
    for t in cps:
        r = agg[t]
        print(f"t={t:6d} dual-sim rec={r['dsim_rec']['mean']:.3f} stale={r['dsim_stale']['mean']:.3f} | "
              f"dual-rec stale={r['drec_stale']['mean']:.3f} | "
              f"tri rec={r['tri_rec']['mean']:.3f} stale={r['tri_stale']['mean']:.3f}")


if __name__ == "__main__":
    main()

# HEMA-2

**A Consolidation-Aware Tri-Memory Architecture with Multi-Channel Scheduling
for Lifelong Conversational AI** — reference implementation.

HEMA-2 extends the dual-memory HEMA architecture (Compact Memory + Vector
Memory, KJAI 2025) into a tri-memory system that can *revise* and *recover*
memories, not just accumulate them — targeting sub-10B on-device models
running under a fixed 3.5K-token prompt budget with frozen weights.

## Why

Append-only conversational memory degrades structurally. In our 10K-turn
memory-layer simulation (components idealized *in favor of* the baseline):

| Turns | Dual-memory recall | Dual-memory stale rate | Tri-memory recall | Tri-memory stale rate |
|------:|---:|---:|---:|---:|
| 100    | 1.00 | 0.00 | 1.00 | 0.00 |
| 2,000  | 0.93 | 1.00 | 0.95 | 0.05 |
| 10,000 | 0.53 | 1.00 | 0.93 | 0.07 |

Old and new values of a fact are near-duplicates in embedding space, so
similarity retrieval drowns the latest value as updates accumulate — while
the tri-memory fact store stays flat at the extractor error bound.
Reproduce in seconds, no model or dependencies required:

```bash
python eval/simulate.py --turns 10000 --seeds 10
```

## Mechanisms (paper section in parentheses)

- **Semantic Fact Store** (`hema2/memory/fact_store.py`, Sec. III-C) —
  time-stamped facts with `ADD` / `MERGE` / `INVALIDATE`; superseded facts
  are marked, never deleted, so temporal queries ("where did I live before
  the move?") remain answerable.
- **Multi-channel scheduler** (`hema2/scheduler/budget.py`, Sec. III-E) —
  splits the memory token budget across recency / similarity / stochastic
  replay / salience-pinned channels, carrying unused budget forward.
- **Multi-query RRF retrieval** (`hema2/retrieval/fusion.py`, Eq. 4) —
  the last N turns each issue a query with position decay, fused by
  reciprocal rank fusion.
- **Provenance-linked rehydration** (`hema2/retrieval/rehydration.py`,
  Sec. III-D) — every summary keeps its source-turn IDs and can expand back
  to verbatim dialogue within budget; edited history invalidates derived
  memories in one pass.
- **Consolidation triggers** (`hema2/consolidation/triggers.py`) — turn-based,
  capacity-based, and hybrid policies.
- **Evaluation suite** (`eval/`) — stale-fact rate, update accuracy,
  abstention F1, degradation slope; E1 knowledge-update stress generator.

## Quick start

```bash
python tests/test_all.py                          # unit tests, no deps
python eval/simulate.py --turns 10000 --seeds 10  # paper Sec. V results
```

Requires Python 3.10+. The core has **zero external dependencies**.

## Running with a local model

The LLM adapter targets any OpenAI-compatible endpoint. Point it at a local
runtime such as **LM Studio** or **Ollama** and the full protocol runs on a
single consumer machine with a sub-10B open-weight model (e.g., Gemma 4 E4B)
— no frontier API access required.

## Status

Mechanism implementations, the degradation simulation, and the E1 generator
are complete. Benchmark adapters (LongMemEval, LoCoMo) and the LLM-in-the-loop
harness are being added as protocol execution proceeds (paper Sec. VI).

## Citation

```bibtex
@inproceedings{ahn2026hema2,
  author    = {Ahn, Kwangseob and Song, Yongjoo},
  title     = {{HEMA-2}: A Consolidation-Aware Tri-Memory Architecture with
               Multi-Channel Scheduling for Lifelong Conversational AI},
  booktitle = {Proceedings of the Korean Artificial Intelligence Association
               Summer Conference},
  year      = {2026}
}

@article{ahn2025hema,
  author  = {Ahn, Kwangseob and Song, Yongjoo},
  title   = {{HEMA}: A Hippocampus-Inspired Extended Memory Architecture for
             Long-Context {AI} Conversations},
  journal = {Korean Journal of Artificial Intelligence},
  volume  = {13}, number = {2}, pages = {1--7}, year = {2025}
}
```

## License

Apache-2.0

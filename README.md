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

```bash
python eval/simulate.py --turns 10000 --seeds 10
```

## Mechanisms

- **Semantic Fact Store** (`hema2/memory/fact_store.py`) — time-stamped facts
  with `ADD` / `MERGE` / `INVALIDATE`; superseded facts are marked, never
  deleted, so temporal queries remain answerable.
- **Multi-channel scheduler** (`hema2/scheduler/budget.py`) — splits a fixed
  memory token budget across recency, similarity, stochastic replay, and
  salience-pinned channels, with carryover of unused allocation.
- **Multi-query RRF retrieval** (`hema2/retrieval/fusion.py`) — fuses searches
  issued by the last N turns with position decay.
- **Provenance-linked rehydration** (`hema2/retrieval/rehydration.py`) — expands
  selected summaries back to source turns within budget and supports one-pass
  invalidation after edited history.
- **Consolidation triggers** (`hema2/consolidation/triggers.py`) — turn-based,
  capacity-based, and hybrid policies.

## Frontier-to-device evaluation

The cross-model harness compares the same memory conditions across hosted
frontier APIs and local Ollama models:

- `no-memory`
- `dual-append`: all old and new assertions coexist
- `tri-fact`: one current valid fact plus explicitly invalidated history

Stable experiment profile names are separated from provider model IDs. This is
important because frontier aliases and preview IDs change over time. Override
IDs in environment variables rather than modifying evaluation code.

```bash
cp .env.example .env

# Local on-device-class run through Ollama
ollama pull qwen3:8b
python eval/run_cross_model.py \
  --profiles ollama \
  --model qwen3:8b \
  --conditions no-memory dual-append tri-fact \
  --seeds 3

# Hosted profiles; each profile can point to its current provider deployment ID
python eval/run_cross_model.py \
  --profiles gpt-5.6 fable-5 kimi-k3 \
  --conditions dual-append tri-fact \
  --seeds 3
```

The core and adapters use only the Python standard library. The Ollama adapter
uses `/api/chat`; hosted profiles use an OpenAI-compatible
`/chat/completions` endpoint.

### Harness evolution and test-time adaptation

Every output row records a frozen harness manifest containing the harness
version, system-prompt hash, temperature, token limit, seed, memory condition,
provider profile, and concrete provider model ID. This prevents prompt or
harness changes from being mistaken for model or memory improvements.

An opt-in `self-guide` condition asks each model to produce a compact set of
memory-reading rules before evaluation. It is recorded separately and must not
be pooled with the non-adapted baseline.

```bash
python eval/run_cross_model.py \
  --profiles ollama \
  --model qwen3:8b \
  --adaptation self-guide
```

This is an evaluation-time adaptation hook, not weight training. Future
self-guided test-time training implementations should be added as new,
versioned adaptation policies rather than silently changing the baseline.

### Output metrics

The answer-level evaluator distinguishes:

- `correct_latest`
- `mixed_latest_and_stale`
- `stale_only`
- `abstained`
- `unsupported_other`

This avoids treating a historically correct mention of an old value as exactly
the same failure as asserting that value as the current fact. Raw answers,
latency, provider token counts, model IDs, seeds, and manifests are retained in
JSONL for later human or judge-model analysis.

## Quick start

```bash
python tests/test_all.py
python eval/simulate.py --turns 10000 --seeds 10
```

Requires Python 3.10+. The core has **zero external dependencies**.

## Status

The memory mechanisms, degradation simulation, E1 knowledge-update generator,
provider-neutral model adapters, Ollama runner, versioned cross-model harness,
and answer-level temporal metrics are implemented. LongMemEval and LoCoMo
adapters, extractor post-verification, full component ablations, and deployment
overhead measurement remain in progress.

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

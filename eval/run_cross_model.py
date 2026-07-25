"""Run HEMA-2 knowledge-update probes across frontier and local models.

Examples:
  python eval/run_cross_model.py --profiles gpt-5.6 kimi-k3 --conditions dual-append tri-fact
  python eval/run_cross_model.py --profiles ollama --model qwen3:8b --conditions no-memory dual-append tri-fact

Provider model aliases are configurable through environment variables; see
README. Results are JSONL and include a frozen harness manifest for auditability.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
from dataclasses import asdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.harness import HarnessManifest, MemoryCase, make_request, save_jsonl
from eval.llm_metrics import aggregate_classifications, classify_current_answer
from eval.synthetic.update_stress import generate
from hema2.models.base import ModelRequest
from hema2.models.registry import create_model, known_profiles

_FACT_RE = re.compile(r"my (?P<attribute>\w+) is now (?P<value>[^.]+)", re.IGNORECASE)


def collect_assertions(turns) -> dict[str, list[tuple[int, str]]]:
    assertions: dict[str, list[tuple[int, str]]] = {}
    for turn in turns:
        if turn.role != "user":
            continue
        match = _FACT_RE.search(turn.text)
        if match:
            assertions.setdefault(match.group("attribute"), []).append(
                (turn.index, match.group("value").strip())
            )
    return assertions


def build_memory(condition: str, attribute: str, assertions: list[tuple[int, str]]) -> str:
    if condition == "no-memory":
        return ""
    if condition == "dual-append":
        return "\n".join(
            f"Episode turn {turn}: user.{attribute} = {value}" for turn, value in assertions
        )
    if condition == "tri-fact":
        turn, value = assertions[-1]
        history = ", ".join(f"{v} (until turn {assertions[i + 1][0]})" for i, (_, v) in enumerate(assertions[:-1]))
        history_line = f"\nHistorical invalidated values: {history}" if history else ""
        return f"VALID fact from turn {turn}: user.{attribute} = {value}{history_line}"
    raise ValueError(f"Unknown condition: {condition}")


def make_cases(n_updates: int, filler_between: int, seed: int, conditions: list[str]) -> list[MemoryCase]:
    turns, probes = generate(n_updates=n_updates, filler_between=filler_between, seed=seed)
    assertions = collect_assertions(turns)
    cases: list[MemoryCase] = []
    for probe in probes:
        attr_assertions = assertions[probe.attribute]
        for condition in conditions:
            cases.append(
                MemoryCase(
                    condition=condition,
                    attribute=probe.attribute,
                    latest=probe.latest,
                    stale=probe.stale,
                    memory_text=build_memory(condition, probe.attribute, attr_assertions),
                    question=f"What is my current {probe.attribute}?",
                    turn_count=probe.asked_at,
                )
            )
    return cases


def create_self_guide(model, manifest: HarnessManifest) -> str:
    request = ModelRequest(
        messages=[
            {
                "role": "system",
                "content": "Create a compact evaluation-time guide. Do not answer any user-memory question.",
            },
            {
                "role": "user",
                "content": (
                    "Write at most four short rules for answering current-state questions from memory records "
                    "that may contain superseded facts. Prefer explicit VALID/latest facts, treat invalidated "
                    "facts as history, abstain when unsupported, and avoid repeating old values unless asked."
                ),
            },
        ],
        temperature=0.0,
        max_tokens=128,
        seed=manifest.seed,
        metadata={"harness_version": manifest.version, "purpose": "self-guide"},
    )
    return model.complete(request).text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", default=["ollama"], choices=known_profiles())
    parser.add_argument("--model", help="Override provider model ID; intended for single-profile runs")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["no-memory", "dual-append", "tri-fact"],
        choices=["no-memory", "dual-append", "tri-fact"],
    )
    parser.add_argument("--adaptation", choices=["none", "self-guide"], default="none")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--filler-between", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--out", default="cross_model_results.jsonl")
    args = parser.parse_args()

    if args.model and len(args.profiles) != 1:
        parser.error("--model can only be used with one --profiles entry")

    rows: list[dict] = []
    for profile in args.profiles:
        model = create_model(profile, args.model)
        manifest = HarnessManifest(max_tokens=args.max_tokens, adaptation=args.adaptation)
        guide = create_self_guide(model, manifest) if args.adaptation == "self-guide" else None
        for seed in range(args.seeds):
            cases = make_cases(args.updates, args.filler_between, seed, args.conditions)
            for case_index, case in enumerate(cases):
                response = model.complete(make_request(case, manifest, guide=guide))
                label = classify_current_answer(response.text, case.latest, case.stale)
                rows.append(
                    {
                        "profile": profile,
                        "provider": response.provider,
                        "model": response.model,
                        "seed": seed,
                        "case_index": case_index,
                        "manifest": manifest.to_dict(),
                        "condition": case.condition,
                        "attribute": case.attribute,
                        "latest": case.latest,
                        "stale": sorted(case.stale),
                        "answer": response.text,
                        "classification": label,
                        "latency_s": response.latency_s,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "self_guide": guide,
                    }
                )
                print(f"{profile:10s} {case.condition:11s} {case.attribute:8s} -> {label}")

    save_jsonl(args.out, rows)
    print(f"\nSaved {len(rows)} rows to {args.out}")
    for profile in args.profiles:
        for condition in args.conditions:
            subset = [r for r in rows if r["profile"] == profile and r["condition"] == condition]
            metrics = aggregate_classifications([r["classification"] for r in subset])
            latencies = [r["latency_s"] for r in subset]
            metrics["mean_latency_s"] = statistics.mean(latencies) if latencies else 0.0
            print(json.dumps({"profile": profile, "condition": condition, **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()

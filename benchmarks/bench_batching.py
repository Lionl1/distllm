"""Throughput of batched decode on one stage: N concurrent sessions, batched vs one by one.

In-process (no relay), so it isolates the compute gain of handle_batch():

    uv run python benchmarks/bench_batching.py --model-id HuggingFaceTB/SmolLM2-360M --device cpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from distllm.hf import load_stage


def run(stage, sessions: int, steps: int, batched: bool) -> float:
    prompt = torch.randint(0, 1000, (1, 32))
    for i in range(sessions):
        stage.handle({"session_id": f"s{i}", "start_pos": 0, "input_ids": prompt})
    started = time.perf_counter()
    for step in range(steps):
        payloads = [
            {"session_id": f"s{i}", "start_pos": 32 + step, "input_ids": torch.tensor([[step + 1]])}
            for i in range(sessions)
        ]
        if batched:
            stage.handle_batch(payloads)
        else:
            for p in payloads:
                stage.handle(p)
    elapsed = time.perf_counter() - started
    stage.drop_sessions()
    return sessions * steps / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--sessions", default="1,2,4,8")
    parser.add_argument("--output", default=str(Path(__file__).with_name("results_batching.md")))
    args = parser.parse_args()

    from transformers import AutoConfig

    n_layers = AutoConfig.from_pretrained(args.model_id).num_hidden_layers
    stage = load_stage(args.model_id, 0, n_layers - 1, device=args.device, dtype=args.dtype)
    run(stage, 2, 2, True)  # warm-up
    lines = [
        "# Batched decode on one stage",
        "",
        f"- model: `{args.model_id}` (all {n_layers} layers in one stage), {stage.dtype} on {stage.device}",
        f"- {args.steps} decode steps per session after a 32-token prompt; tokens/s summed over sessions",
        "",
        "| sessions | one by one, tok/s | batched, tok/s | speed-up |",
        "|---:|---:|---:|---:|",
    ]
    for n in [int(x) for x in args.sessions.split(",")]:
        single = run(stage, n, args.steps, batched=False)
        batched = run(stage, n, args.steps, batched=True)
        lines.append(f"| {n} | {single:.1f} | {batched:.1f} | {batched / single:.2f}x |")
        print(lines[-1])
    report = "\n".join(lines) + "\n"
    Path(args.output).write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()

"""Memory and quality of quantized stages (bitsandbytes), in-process, no relay.

For each method the model is split into --stages stages; we report the weight memory
of the largest stage, greedy-token agreement with the float32 pipeline and the time
per token.

    uv run --extra quantization python benchmarks/bench_quantization.py --model-id Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from pathlib import Path

import torch

from distllm.hf import PipelineGenerator, load_stage

PROMPT = (
    "Distributed inference splits a large language model across several machines. "
    "Each machine keeps a slice of the transformer layers and passes hidden states "
    "to the next one. The main advantages of this approach are"
)


def _stage_bytes(stage, blocks_only: bool = False) -> int:
    module = stage.base.layers if blocks_only else stage.model
    tensors = list(module.parameters()) + list(module.buffers())
    seen, total = set(), 0
    for t in tensors:
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total


async def _run(stages, input_ids, new_tokens):
    by_role = {f"stage-{i}": s for i, s in enumerate(stages)}

    async def call(role, payload):
        return by_role[role].handle(payload)

    return await PipelineGenerator(call, list(by_role)).generate(input_ids, new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--methods", default="none,int8,nf4,fp4")
    parser.add_argument("--output", default=str(Path(__file__).with_name("results_quantization.md")))
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    input_ids = AutoTokenizer.from_pretrained(args.model_id)(PROMPT, return_tensors="pt").input_ids
    n_layers = AutoConfig.from_pretrained(args.model_id).num_hidden_layers
    bounds = [round(i * n_layers / args.stages) for i in range(args.stages + 1)]

    rows, reference = [], None
    for method in [m.strip() for m in args.methods.split(",")]:
        quantize = None if method == "none" else method
        print(f"[*] {method} ...")
        stages = [
            load_stage(args.model_id, bounds[i], bounds[i + 1] - 1, device="cpu", dtype="float32", quantize=quantize)
            for i in range(args.stages)
        ]
        result = asyncio.run(_run(stages, input_ids, args.new_tokens))
        if reference is None:
            reference = result.token_ids
        same = sum(a == b for a, b in zip(result.token_ids, reference))
        first_diff = next((i for i, (a, b) in enumerate(zip(result.token_ids, reference)) if a != b), None)
        rows.append((
            method,
            sum(_stage_bytes(s, blocks_only=True) for s in stages) / 2**20,
            max(_stage_bytes(s) for s in stages) / 2**20,
            same / len(reference),
            first_diff,
            statistics.median(result.token_seconds) * 1000,
        ))
        del stages

    lines = [
        "# Quantized stages",
        "",
        f"- model: `{args.model_id}` ({n_layers} layers), {args.stages} stages, CPU, compute dtype float32",
        f"- {args.new_tokens} greedy tokens; agreement = share of positions equal to the float32 pipeline "
        "(greedy decoding diverges for good after the first different token)",
        "- embeddings / lm_head are not quantized; they dominate the first/last stage of small models",
        "",
        "| method | transformer blocks, MiB | largest stage, MiB | token agreement with fp32 | first differing token | per token, ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, blocks, mib, agree, first_diff, ms in rows:
        lines.append(
            f"| {method} | {blocks:.0f} | {mib:.0f} | {agree:.0%} | {'—' if first_diff is None else first_diff} | {ms:.0f} |"
        )
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    Path(args.output).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

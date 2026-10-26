"""End-to-end benchmark: relay + N stage workers as separate processes on this machine.

Compares
  * legacy      — old behaviour: no KV cache (whole sequence recomputed every token),
                  client short-polls results every 1s, worker sleeps 0.5s between polls;
  * no-kv-cache — long-poll, but still recomputing the whole sequence;
  * distllm     — long-poll + per-session KV cache on the stages (current default);
  * single process HF model with KV cache — the lower bound without any networking.
It also checks that the pipeline produces exactly the same tokens as the full model
and measures the size of one hidden-state transfer in the old and new wire format.

Usage:
    uv run python benchmarks/bench_pipeline.py --model-id Qwen/Qwen2.5-0.5B --new-tokens 32
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import platform
import socket
import statistics
import time
from pathlib import Path

import msgpack
import torch

PROMPT = (
    "Distributed inference splits a large language model across several machines. "
    "Each machine keeps a slice of the transformer layers and passes hidden states "
    "to the next one. The main advantages of this approach are"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_relay(port: int) -> None:
    import uvicorn

    from distllm.relay import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _run_stage_worker(model_id, layer_start, layer_end, role, relay_url, device, dtype, idle_sleep) -> None:
    import builtins

    from distllm import worker_node
    from distllm.hf import load_stage

    builtins.print = lambda *a, **k: None  # worker logs would drown the report
    stage = load_stage(model_id, layer_start, layer_end, device=device, dtype=dtype)

    @worker_node(role=role, relay_url=relay_url)
    def process(payload):
        return stage.handle(payload)

    process.start(node_id=role, idle_sleep=idle_sleep)


def _split(n_layers: int, n_stages: int) -> list[tuple[int, int]]:
    bounds = [round(i * n_layers / n_stages) for i in range(n_stages + 1)]
    return [(bounds[i], bounds[i + 1] - 1) for i in range(n_stages)]


async def _wait_for_workers(relay_url: str, roles: list[str], timeout: float = 600.0) -> None:
    """Workers are ready once a close-session probe round-trips through every role."""
    from distllm import Cluster

    async with Cluster(relay_url) as cluster:
        deadline = time.monotonic() + timeout
        for role in roles:
            while True:
                try:
                    await cluster.call(role, {"session_id": "probe", "close": True}, timeout=5)
                    break
                except Exception:  # noqa: BLE001 — not up yet
                    if time.monotonic() > deadline:
                        raise
                    await asyncio.sleep(1)


async def _pipeline_run(relay_url, roles, input_ids, new_tokens, *, use_kv_cache, poll_interval):
    from distllm import Cluster
    from distllm.hf import PipelineGenerator

    async with Cluster(relay_url, poll_interval=poll_interval) as cluster:
        generator = PipelineGenerator(cluster.call, roles, use_kv_cache=use_kv_cache)
        return await generator.generate(input_ids, new_tokens)


def _single_process(model_id, input_ids, new_tokens, device, dtype):
    from transformers import AutoModelForCausalLM, DynamicCache

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype).to(device).eval()
    cache = DynamicCache(config=model.config)
    ids = input_ids.to(device)
    tokens, times = [], []
    with torch.inference_mode():
        nxt = ids
        for step in range(new_tokens):
            t0 = time.perf_counter()
            logits = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1]
            token = int(logits.argmax(-1))
            times.append(time.perf_counter() - t0)
            tokens.append(token)
            nxt = torch.tensor([[token]], device=device)
    return tokens, times[0], times[1:], model.config


def _legacy_size(tensor: torch.Tensor) -> int:
    import msgpack_numpy

    msgpack_numpy.patch()
    arr = tensor.float().numpy()  # old format: everything through float32 numpy
    return len(msgpack.packb({"hidden_states": {"__tensor__": True, "arr": arr, "is_bf16": True}}, use_bin_type=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--modes", default="distllm,no-kv-cache,legacy")
    parser.add_argument("--output", default=str(Path(__file__).with_name("results.md")))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from distllm.transport import pack_payload

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    print(f"[*] Single-process reference: {args.model_id}, {args.dtype} on {args.device}")
    ref_tokens, ref_prefill, ref_times, config = _single_process(
        args.model_id, input_ids, args.new_tokens, args.device, args.dtype
    )
    splits = _split(config.num_hidden_layers, args.stages)
    roles = [f"stage-{i}" for i in range(len(splits))]

    rows = []
    rows.append(("single process (no network)", ref_prefill, ref_times, True, 0))

    modes = {
        "distllm": dict(use_kv_cache=True, poll_interval=None, idle_sleep=0.0,
                        label="distllm: long-poll + KV cache"),
        "no-kv-cache": dict(use_kv_cache=False, poll_interval=None, idle_sleep=0.0,
                            label="long-poll, no KV cache"),
        "legacy": dict(use_kv_cache=False, poll_interval=1.0, idle_sleep=0.5,
                       label="legacy: 1s polling, 0.5s worker sleep, no KV cache"),
    }
    ctx = mp.get_context("spawn")
    for key in [m.strip() for m in args.modes.split(",") if m.strip()]:
        mode = modes[key]
        port = _free_port()
        relay_url = f"http://127.0.0.1:{port}"
        procs = [ctx.Process(target=_run_relay, args=(port,), daemon=True)]
        for role, (a, b) in zip(roles, splits):
            procs.append(ctx.Process(
                target=_run_stage_worker,
                args=(args.model_id, a, b, role, relay_url, args.device, args.dtype, mode["idle_sleep"]),
                daemon=True,
            ))
        for p in procs:
            p.start()
        try:
            asyncio.run(_wait_for_workers(relay_url, roles))
            print(f"[*] {mode['label']} ...")
            result = asyncio.run(_pipeline_run(
                relay_url, roles, input_ids, args.new_tokens,
                use_kv_cache=mode["use_kv_cache"], poll_interval=mode["poll_interval"],
            ))
        finally:
            for p in procs:
                p.terminate()
            for p in procs:
                p.join(timeout=10)
        rows.append((mode["label"], result.prefill_seconds, result.token_seconds,
                     result.token_ids == ref_tokens, result.cache_recoveries))

    # One hidden-state hop for the whole prompt, in half precision as on a GPU node.
    hidden = torch.randn(1, prompt_len, config.hidden_size, dtype=torch.bfloat16)
    new_size = len(pack_payload({"hidden_states": hidden}))
    old_size = _legacy_size(hidden)

    lines = [
        f"# distllm benchmark",
        "",
        f"- model: `{args.model_id}` ({config.num_hidden_layers} layers), {args.dtype} on {args.device}",
        f"- stages: {len(splits)} ({', '.join(f'{a}-{b}' for a, b in splits)}), relay and workers are "
        f"separate processes on one machine ({platform.processor() or platform.machine()}, {platform.system()})",
        f"- prompt: {prompt_len} tokens, generated: {args.new_tokens} tokens, greedy",
        "",
        "| mode | prefill, ms | per token, ms (median) | tokens/s | same tokens as full model |",
        "|---|---:|---:|---:|:---:|",
    ]
    for label, prefill, times, same, recoveries in rows:
        median = statistics.median(times) * 1000 if times else float("nan")
        tps = len(times) / sum(times) if times else 0.0
        lines.append(f"| {label} | {prefill * 1000:.0f} | {median:.0f} | {tps:.2f} | {'yes' if same else 'NO'} |")
    lines += [
        "",
        f"Hidden-state transfer for the {prompt_len}-token prompt "
        f"(bf16, hidden size {config.hidden_size}): legacy format {old_size / 1024:.0f} KiB, "
        f"new format {new_size / 1024:.0f} KiB ({old_size / new_size:.1f}x smaller).",
    ]
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"[v] Saved to {args.output}")


if __name__ == "__main__":
    main()

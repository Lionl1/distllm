"""End-to-end benchmark: relay + N stage workers as separate processes on this machine.

Modes (pick with --modes):
  * legacy       — old behaviour: no KV cache (whole sequence recomputed every token),
                   client short-polls results every 1s, worker sleeps 0.5s between polls;
  * no-kv-cache  — long-poll, but still recomputing the whole sequence;
  * distllm      — long-poll + per-session KV cache, one client request per stage;
  * route        — same, but the relay chains the stages: one client request per token;
  * spec-K       — speculative decoding with --draft-model-id over routes, K draft tokens
                   per round (e.g. spec-2,spec-4,spec-6);
plus a single-process HF model with KV cache — the lower bound without any networking.

--worker-rtt-ms / --client-rtt-ms make the relay add an artificial round trip to every
request (see distllm.relay.set_simulated_latency) to model a real network.
Every mode is checked to produce exactly the tokens of the full model.

Usage:
    uv run python benchmarks/bench_pipeline.py --model-id Qwen/Qwen2.5-0.5B --modes distllm,no-kv-cache,legacy
    uv run python benchmarks/bench_pipeline.py --model-id HuggingFaceTB/SmolLM2-360M \\
        --draft-model-id HuggingFaceTB/SmolLM2-135M --worker-rtt-ms 60 --client-rtt-ms 20 \\
        --modes distllm,route,spec-2,spec-4,spec-6
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


def _run_relay(port: int, worker_rtt_ms: float, client_rtt_ms: float) -> None:
    import logging

    from distllm.relay import start_relay

    logging.getLogger("uvicorn.access").disabled = True
    start_relay(port=port, worker_rtt_ms=worker_rtt_ms, client_rtt_ms=client_rtt_ms, host="127.0.0.1")


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


async def _pipeline_run(relay_url, roles, input_ids, new_tokens, mode, draft):
    from distllm import Cluster
    from distllm.hf import PipelineGenerator, SpeculativeGenerator

    async with Cluster(relay_url, poll_interval=mode["poll_interval"]) as cluster:
        route_call = cluster.call_route if mode["route"] else None
        if mode["k"]:
            generator = SpeculativeGenerator(cluster.call, roles, draft, k=mode["k"], route_call=route_call)
        else:
            generator = PipelineGenerator(
                cluster.call, roles, use_kv_cache=mode["use_kv_cache"], route_call=route_call
            )
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
        for _ in range(new_tokens):
            t0 = time.perf_counter()
            logits = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1]
            token = int(logits.argmax(-1))
            times.append(time.perf_counter() - t0)
            tokens.append(token)
            nxt = torch.tensor([[token]], device=device)
    config = model.config
    del model, cache
    return tokens, times[0], times[1:], config


def _legacy_size(tensor: torch.Tensor) -> int:
    import msgpack_numpy

    msgpack_numpy.patch()
    arr = tensor.float().numpy()  # old format: everything through float32 numpy
    return len(msgpack.packb({"hidden_states": {"__tensor__": True, "arr": arr, "is_bf16": True}}, use_bin_type=True))


def _mode(key: str) -> dict:
    base = dict(use_kv_cache=True, poll_interval=None, idle_sleep=0.0, route=False, k=0)
    if key == "distllm":
        return {**base, "label": "long-poll + KV cache, request per stage"}
    if key == "route":
        return {**base, "route": True, "label": "+ relay-side route, request per token"}
    if key == "no-kv-cache":
        return {**base, "use_kv_cache": False, "label": "long-poll, no KV cache"}
    if key == "legacy":
        return {**base, "use_kv_cache": False, "poll_interval": 1.0, "idle_sleep": 0.5,
                "label": "legacy: 1s polling, 0.5s worker sleep, no KV cache"}
    if key.startswith("spec-"):
        k = int(key.split("-", 1)[1])
        return {**base, "route": True, "k": k, "label": f"+ speculative decoding, k={k}"}
    raise ValueError(f"unknown mode {key!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--draft-model-id", default=None, help="draft model for spec-K modes")
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--worker-rtt-ms", type=float, default=0.0)
    parser.add_argument("--client-rtt-ms", type=float, default=0.0)
    parser.add_argument("--modes", default="distllm,no-kv-cache,legacy")
    parser.add_argument("--relay-bin", default=None, help="use the Rust relay binary instead of the Python relay")
    parser.add_argument("--output", default=str(Path(__file__).with_name("results.md")))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from distllm.transport import pack_payload

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    mode_keys = [m.strip() for m in args.modes.split(",") if m.strip()]
    modes = {key: _mode(key) for key in mode_keys}

    draft = None
    if any(m["k"] for m in modes.values()):
        if not args.draft_model_id:
            parser.error("spec-K modes need --draft-model-id")
        draft = AutoModelForCausalLM.from_pretrained(args.draft_model_id, dtype=torch.float32)

    print(f"[*] Single-process reference: {args.model_id}, {args.dtype} on {args.device}")
    ref_tokens, ref_prefill, ref_times, config = _single_process(
        args.model_id, input_ids, args.new_tokens, args.device, args.dtype
    )
    splits = _split(config.num_hidden_layers, args.stages)
    roles = [f"stage-{i}" for i in range(len(splits))]

    rows = [("single process (no network)", ref_prefill, ref_times, True, "")]
    ctx = mp.get_context("spawn")
    for key, mode in modes.items():
        port = _free_port()
        relay_url = f"http://127.0.0.1:{port}"
        relay_proc = None
        if args.relay_bin:
            import subprocess

            relay_proc = subprocess.Popen(
                [args.relay_bin, "--port", str(port), "--host", "127.0.0.1",
                 "--simulate-worker-rtt-ms", str(args.worker_rtt_ms),
                 "--simulate-client-rtt-ms", str(args.client_rtt_ms)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs = []
        else:
            procs = [ctx.Process(target=_run_relay, args=(port, args.worker_rtt_ms, args.client_rtt_ms), daemon=True)]
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
            result = asyncio.run(_pipeline_run(relay_url, roles, input_ids, args.new_tokens, mode, draft))
        finally:
            for p in procs:
                p.terminate()
            for p in procs:
                p.join(timeout=10)
            if relay_proc is not None:
                relay_proc.terminate()
                relay_proc.wait(timeout=10)
        extra = ""
        if hasattr(result, "acceptance_rate"):
            extra = f"acceptance {result.acceptance_rate:.0%}, {result.tokens_per_round:.1f} tokens/round"
        rows.append((mode["label"], result.prefill_seconds, result.token_seconds,
                     result.token_ids == ref_tokens, extra))

    # One hidden-state hop for the whole prompt, in half precision as on a GPU node.
    hidden = torch.randn(1, prompt_len, config.hidden_size, dtype=torch.bfloat16)
    new_size = len(pack_payload({"hidden_states": hidden}))
    old_size = _legacy_size(hidden)

    network = (
        f"simulated RTT to relay: workers {args.worker_rtt_ms:.0f} ms, client {args.client_rtt_ms:.0f} ms"
        if args.worker_rtt_ms or args.client_rtt_ms else "no simulated latency (loopback)"
    )
    lines = [
        "# distllm benchmark",
        "",
        f"- model: `{args.model_id}` ({config.num_hidden_layers} layers), {args.dtype} on {args.device}"
        + (f"; draft: `{args.draft_model_id}`" if draft is not None else ""),
        f"- stages: {len(splits)} ({', '.join(f'{a}-{b}' for a, b in splits)}), relay and workers are "
        f"separate processes on one machine ({platform.processor() or platform.machine()}, {platform.system()})",
        f"- network: {network}; relay: {'Rust (' + args.relay_bin + ')' if args.relay_bin else 'Python'}",
        f"- prompt: {prompt_len} tokens, generated: {args.new_tokens} tokens, greedy",
        "",
        "| mode | prefill, ms | per token, ms (median) | tokens/s | same tokens as full model | notes |",
        "|---|---:|---:|---:|:---:|---|",
    ]
    for label, prefill, times, same, extra in rows:
        median = statistics.median(times) * 1000 if times else float("nan")
        tps = len(times) / sum(times) if times else 0.0
        lines.append(
            f"| {label} | {prefill * 1000:.0f} | {median:.0f} | {tps:.2f} | {'yes' if same else 'NO'} | {extra} |"
        )
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

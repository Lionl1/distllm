# Benchmarks

All numbers were measured on one machine (Apple Silicon laptop, 8 GB RAM; CPU unless a
section says MPS):
the relay and every stage run as **separate processes** talking HTTP over loopback, the
client drives generation through them, and every run is checked to produce **exactly the
same tokens** as the full model in one process (except quantized runs, which are compared
with float32). Network latency, where shown, is simulated by the relay
(`--worker-rtt-ms/--client-rtt-ms`). CUDA numbers are not available yet.

Scripts: [`benchmarks/bench_pipeline.py`](../benchmarks/bench_pipeline.py) (pipeline modes,
latency simulation, speculative decoding) and
[`benchmarks/bench_quantization.py`](../benchmarks/bench_quantization.py) (memory and quality
of quantized stages), [`benchmarks/bench_batching.py`](../benchmarks/bench_batching.py)
(batched decode) and [`benchmarks/bench_relay.py`](../benchmarks/bench_relay.py) (Python vs
Rust relay). Raw outputs are in `benchmarks/results*.md`.

What the modes mean:

- **legacy** — the behaviour before 0.2: no KV cache (the whole sequence is recomputed for
  every token), the client polls results every second, the worker sleeps 0.5 s between polls;
- **long-poll + KV cache** — one client request per stage per token;
- **route** — the relay chains the stages: one client request per token;
- **speculative, k** — the draft model proposes *k* tokens per round, verified in one pass.

## Pipeline vs. old behaviour

- model: `Qwen/Qwen2.5-0.5B` (24 layers), float32 on cpu
- stages: 2 (0-11, 12-23), relay and workers are separate processes on one machine (arm, Darwin)
- prompt: 37 tokens, generated: 32 tokens, greedy

| mode | prefill, ms | per token, ms (median) | tokens/s | same tokens as full model |
|---|---:|---:|---:|:---:|
| single process (no network) | 659 | 69 | 13.38 | yes |
| distllm: long-poll + KV cache | 578 | 79 | 10.86 | yes |
| long-poll, no KV cache | 518 | 209 | 4.57 | yes |
| legacy: 1s polling, 0.5s worker sleep, no KV cache | 2041 | 2026 | 0.49 | yes |

Hidden-state transfer for the 37-token prompt (bf16, hidden size 896): legacy format 130 KiB, new format 65 KiB (2.0x smaller).

On a real network every hop adds round-trip latency; the speed-up comes from removing the polling waits (≈1.5 s per hop) and recomputation (O(n²) → O(n)), which grow with distance and length.

Reproduce: `uv run python benchmarks/bench_pipeline.py --model-id Qwen/Qwen2.5-0.5B`.

## Beating network latency

On loopback a hop costs ~1 ms. Over a real network (free Colab GPUs behind a tunnel)
every request to the relay costs a round trip, and that, not GPU time, bounds the
speed: a token through 2 stages with 60 ms RTT is ~250 ms of pure networking.
Two techniques attack it:

- **Relay-side routes** (`PipelineGenerator(..., route_call=cluster.call_route)`): the
  relay forwards hidden states from stage to stage itself; the client sends one
  request per token instead of one per stage. Saves the client legs of every hop —
  the farther the client is from the relay, the bigger the win.
- **Speculative decoding** (`SpeculativeGenerator(call, roles, draft_model, k=4)`): a small
  draft model with the same tokenizer runs on the client and proposes `k` tokens; the
  pipeline verifies all of them in one pass (`output="argmax"`) and rejected positions
  are cropped from the stage caches (`rewind`). One round trip yields up to `k+1`
  tokens, and the output is exactly greedy decoding of the big model.

Simulate a network locally: `distllm relay --simulate-worker-rtt-ms 60 --simulate-client-rtt-ms 20`
(or `--worker-rtt-ms/--client-rtt-ms` in the benchmark).

**With simulated network latency:**

- model: `HuggingFaceTB/SmolLM2-360M` (32 layers), float32 on cpu; draft: `HuggingFaceTB/SmolLM2-135M`
- stages: 2 (0-15, 16-31), relay and workers are separate processes on one machine (arm, Darwin)
- network: simulated RTT to relay: workers 60 ms, client 20 ms
- prompt: 37 tokens, generated: 48 tokens, greedy

| mode | prefill, ms | per token, ms (median) | tokens/s | same tokens as full model | notes |
|---|---:|---:|---:|:---:|---|
| single process (no network) | 481 | 51 | 18.65 | yes |  |
| long-poll + KV cache, request per stage | 591 | 259 | 3.80 | yes |  |
| + relay-side route, request per token | 537 | 229 | 4.35 | yes |  |
| + speculative decoding, k=2 | 698 | 136 | 6.06 | yes | acceptance 74%, 2.4 tokens/round |
| + speculative decoding, k=4 | 663 | 90 | 7.85 | yes | acceptance 58%, 3.2 tokens/round |
| + speculative decoding, k=6 | 691 | 70 | 8.46 | yes | acceptance 45%, 3.7 tokens/round |

Hidden-state transfer for the 37-token prompt (bf16, hidden size 960): legacy format 139 KiB, new format 69 KiB (2.0x smaller).

Over the network the pipeline is ~5x slower than one process — pure round trips.
Routes win ~14% here (the client is close to the relay); speculative decoding wins
**2.2x** (3.80 → 8.46 tokens/s at k=6) with identical output: 3.7 tokens per round trip
even at 45% acceptance.

**Same models, no latency (loopback):**

- model: `HuggingFaceTB/SmolLM2-360M` (32 layers), float32 on cpu; draft: `HuggingFaceTB/SmolLM2-135M`
- stages: 2 (0-15, 16-31), relay and workers are separate processes on one machine (arm, Darwin)
- network: no simulated latency (loopback)
- prompt: 37 tokens, generated: 48 tokens, greedy

| mode | prefill, ms | per token, ms (median) | tokens/s | same tokens as full model | notes |
|---|---:|---:|---:|:---:|---|
| single process (no network) | 190 | 50 | 19.78 | yes |  |
| long-poll + KV cache, request per stage | 323 | 60 | 16.04 | yes |  |
| + relay-side route, request per token | 323 | 58 | 16.57 | yes |  |
| + speculative decoding, k=2 | 551 | 83 | 10.12 | yes | acceptance 74%, 2.4 tokens/round |
| + speculative decoding, k=4 | 516 | 59 | 12.68 | yes | acceptance 58%, 3.2 tokens/round |
| + speculative decoding, k=6 | 550 | 56 | 12.67 | yes | acceptance 45%, 3.7 tokens/round |

Hidden-state transfer for the 37-token prompt (bf16, hidden size 960): legacy format 139 KiB, new format 69 KiB (2.0x smaller).

Without latency speculative decoding *loses* (12.7 vs 16.6 tokens/s): the draft is only
~2.7x smaller than the target and runs on the same CPU, so drafting costs as much as it
saves. It pays off when round trips dominate — exactly the distributed case — and with a
much smaller draft relative to the target (e.g. 0.5B draft for a 7B+ target).

Reproduce:

```bash
uv run python benchmarks/bench_pipeline.py --model-id HuggingFaceTB/SmolLM2-360M \
    --draft-model-id HuggingFaceTB/SmolLM2-135M --new-tokens 48 \
    --worker-rtt-ms 60 --client-rtt-ms 20 --modes distllm,route,spec-2,spec-4,spec-6
```

## Quantized stages

`load_stage(..., quantize="nf4" | "fp4" | "int8")` (worker flag `--quantize`) quantizes the
linear layers of every transformer block with bitsandbytes while loading — layer by layer,
so peak memory is the quantized stage plus one full-precision layer. Embeddings, the final
norm and `lm_head` stay in the stage dtype. Install with `uv sync --extra quantization`
(bitsandbytes >= 0.46 also runs on CPU).

This is what makes free Colab useful: a 7B model in float16 is ~14 GB, in nf4 ~4 GB of
transformer blocks — split over two T4s it leaves plenty of room for the KV cache.

- model: `Qwen/Qwen2.5-0.5B` (24 layers), 2 stages, CPU, compute dtype float32
- 32 greedy tokens; agreement = share of positions equal to the float32 pipeline (greedy decoding diverges for good after the first different token)
- embeddings / lm_head are not quantized; they dominate the first/last stage of small models

| method | transformer blocks, MiB | largest stage, MiB | token agreement with fp32 | first differing token | per token, ms |
|---|---:|---:|---:|---:|---:|
| none | 1365 | 1202 | 100% | — | 62 |
| int8 | 342 | 690 | 19% | 2 | 176 |
| nf4 | 171 | 605 | 3% | 0 | 234 |
| fp4 | 171 | 605 | 19% | 6 | 226 |

The quantized output is identical to quantizing the same model with transformers'
`BitsAndBytesConfig` (checked for nf4 and int8), i.e. the split itself adds no error. The
quality loss is the quantization's: a 0.5B model is very sensitive — greedy decoding
diverges from float32 within a few tokens. Larger models tolerate nf4 much better; measure
on yours: `uv run --extra quantization python benchmarks/bench_quantization.py --model-id ...`.

## Apple Silicon GPU (MPS)

Same laptop, two stages in one process (no relay), 32 greedy tokens after a warm-up, best of
two runs:

| model | CPU float32, tok/s | MPS float16, tok/s | output vs CPU float32 |
|---|---:|---:|---|
| Qwen/Qwen2.5-0.5B | 12.5 | 20.6 | identical (16/16 tokens checked) |
| HuggingFaceTB/SmolLM2-360M | 14.8 | 20.0 | identical (16/16 tokens checked) |

bfloat16 on MPS drifted from float32 after 6–11 tokens, so `dtype="auto"` uses float16 there.

## Batched decode

Several sessions decoding at once on one stage (`benchmarks/bench_batching.py`, in-process,
SmolLM2-360M, all 32 layers in one stage, 16 decode steps per session after a 32-token
prompt; tokens/s summed over sessions):

**MPS, float16**

|sessions | one by one, tok/s | batched, tok/s | speed-up |
|---:|---:|---:|---:|
| 1 | 18.8 | 18.9 | 1.00x |
| 2 | 19.6 | 27.5 | 1.40x |
| 4 | 20.2 | 38.4 | 1.90x |
| 8 | 20.0 | 56.6 | 2.84x |

**CPU, float32**

|sessions | one by one, tok/s | batched, tok/s | speed-up |
|---:|---:|---:|---:|
| 1 | 14.9 | 14.8 | 0.99x |
| 2 | 13.6 | 7.0 | 0.52x |
| 4 | 13.4 | 14.3 | 1.06x |
| 8 | 16.0 | 27.9 | 1.75x |

On CPU, padding and cache copies make a batch of 2 slower than running the sessions one by
one (0.5–0.8× over repeated runs); from 4 sessions batching wins (1.1–2.3×). CPU stages
therefore batch only from 4 sessions (`PipelineStage.min_batch`), GPU/MPS stages from 2.

## Python relay vs Rust relay

Same machine (Apple Silicon laptop, CPU), echo workers in Python, client `Cluster` in Python,
`benchmarks/bench_relay.py`, median of 200 calls (decode) / 50 calls (prefill), two runs each:

| request | Python relay | Rust relay |
|---|---:|---:|
| decode step (1 token, 1.7 KiB), one hop | 4.03–4.05 ms | **1.79–1.81 ms** |
| decode step, 2-stage route | 6.79–6.84 ms | **3.17–3.22 ms** |
| prefill 512 tokens (0.9 MiB), one hop | 9.43–10.48 ms | **6.02–6.16 ms** |
| prefill, 2-stage route | 14.78–15.54 ms | **9.68–9.82 ms** |

Memory after 300 registrations: Python relay 242 MB RSS (it imports torch through the transport
module), Rust relay 3.6 MB.

End to end (SmolLM2-360M, 2 stages, route, CPU, 48 tokens, three interleaved runs) the median
time per token was 79–83 ms with the Python relay and 73–89 ms with the Rust one, against 71 ms
for the model in one process: the relay's 2–3 ms per token are below the noise of a CPU-bound
run on a busy laptop. The Rust relay pays off in per-hop latency under load, memory, and
deployment (one static binary on a small VPS, no Python).

## Against llama.cpp RPC and Petals

Same model (SmolLM2-360M, float32, CPU), same machine, two nodes, 48 tokens
([`benchmarks/compare_llamacpp.sh`](../benchmarks/compare_llamacpp.sh),
[`results_comparison.md`](../benchmarks/results_comparison.md)):

| setup, tokens/s | llama.cpp RPC | distllm |
|---|---:|---:|
| one process | 30.6 | 19.8 |
| 2 nodes, loopback | 35.3 | 16.0 |
| 2 nodes, 60 ms RTT | 6.3 | 3.8 (route 4.4) |
| 2 nodes, 60 ms RTT, speculative k=6 | — | **8.5** |

- **Raw speed:** llama.cpp's ggml kernels are ~1.5x faster than PyTorch on CPU, and over RPC
  it pays about one round trip per node per token — distllm without speculation is slower.
  With speculative decoding over the network distllm is 1.35x faster than llama.cpp RPC at
  60 ms, because it pays round trips per *round*, not per token.
- **A node dies:** llama.cpp RPC aborts the client process (`Remote RPC server crashed`,
  SIGABRT) and the generation is lost; distllm finishes it with identical tokens on another
  worker (the chaos tests). llama.cpp RPC also needs the client to reach every node directly
  (no NAT traversal) and has no authentication; its README warns against exposing it.
- **Petals** 2.2 (last release 2023) needs torch 2.1; on macOS arm64 its inference sessions
  were reset by the server, so there are no numbers yet — to be rerun on Linux.
